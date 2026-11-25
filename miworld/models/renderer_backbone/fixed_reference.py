"""Fixed spatial reference sampling, matching the isolated RGB-readout branch.

No RGB head, target image, GT mask, or GT bbox participates in this module.
Supports all frames/residents and sums per-resident maps in target space.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def relative_camera_pose(cond, target):
    position=cond['player_position']
    idx=torch.arange(len(target),device=position.device)
    camera=(position+cond['camera_relative'])[idx,:,target]
    direction=cond['camera_direction'][idx,:,target]
    yaw=cond['yaw_pitch'][...,0]
    right=torch.stack((yaw.cos(),-yaw.sin(),yaw*0),-1)
    forward=torch.stack((yaw.sin(),yaw.cos(),yaw*0),-1)
    up=torch.zeros_like(right);up[...,2]=1
    basis=torch.stack((right,forward,up),-2)
    delta=camera[:,:,None]-position
    ray=torch.einsum('btaij,btaj->btai',basis,F.normalize(delta,dim=-1,eps=1e-6))
    optical=torch.einsum('btaij,btj->btai',basis,F.normalize(direction,dim=-1,eps=1e-6))
    distance=delta.norm(dim=-1,keepdim=True).log1p()
    fov=cond['fov_x'][idx,:,target,None,None].expand(*distance.shape)
    return torch.cat((ray,optical,distance,fov),-1)


class FixedSpatialReferenceProjector(nn.Module):
    def __init__(self,height=36,width=64,gradient_checkpointing=True,
                 depth_sharpness=8.,output_channels=32,actor_dim=30,appearance_uv_view=False):
        super().__init__()
        self.height,self.width=height,width
        self.value_channels=32
        self.appearance_uv_view=bool(appearance_uv_view)
        self.actor_dim=int(actor_dim)
        if self.actor_dim <= 0:
            raise ValueError("actor_dim must be positive")
        self.output_channels=int(output_channels)
        if self.output_channels <= 0:
            raise ValueError("output_channels must be positive")
        self.gradient_checkpointing=gradient_checkpointing
        self.depth_sharpness=float(depth_sharpness)
        self.norm=nn.LayerNorm(256);self.value=nn.Linear(256,self.value_channels)
        query_channels=2+2+self.actor_dim+8
        self.fuse=nn.Sequential(
            nn.Conv2d(4*self.value_channels+query_channels,64,3,padding=1),
            nn.SiLU(),
            nn.Conv2d(64,self.output_channels,1),
        )
        y,x=torch.meshgrid(torch.linspace(-1,1,height),torch.linspace(-1,1,width),indexing='ij')
        self.register_buffer('screen',torch.stack((x,y),0),persistent=False)

    def _project_impl(self,values,roi,local,actor,valid,pose,appearance_only):
        n=len(values);h,w=self.height,self.width
        dtype=values.dtype
        screen=self.screen[None].expand(n,-1,-1,-1).to(dtype)
        coordinates=local.permute(0,3,1,2).to(dtype)
        camera_pose=pose[:,:,None,None].expand(-1,-1,h,w).to(dtype)
        if appearance_only:
            # The split architecture deliberately exposes only ROI-routed
            # appearance/ID at the image input. Coordinates remain internal to
            # reference sampling; action/state and camera pose use a separate
            # DiT condition route.
            screen=torch.zeros_like(screen)
            if self.appearance_uv_view:
                # Body-local UV plus camera ray/optical direction in actor coordinates.
                # Absolute screen location, distance, FOV and reference RGB remain off.
                camera_pose=torch.cat((camera_pose[:,:6],torch.zeros_like(camera_pose[:,6:])),1)
            else:
                coordinates=torch.zeros_like(coordinates)
                camera_pose=torch.zeros_like(camera_pose)
        query=torch.cat((screen,coordinates,
            actor[:,:,None,None].expand(-1,-1,h,w).to(dtype),camera_pose),1)
        # Match diagnostic autocast behavior, including coordinate rounding.
        grid=local[:,None].expand(-1,4,-1,-1,-1).to(dtype)
        sampled=F.grid_sample(values.flatten(0,1).float(),grid.reshape(n*4,h,w,2).float(),
            align_corners=False,padding_mode='zeros').reshape(
                n,4,self.value_channels,h,w
            ).to(dtype)
        sampled=sampled*valid[:,:,None,None,None].to(dtype)
        if appearance_only:
            # ID-only experiments must not receive reference-encoder bias or
            # position encodings after the image pixels have been disabled.
            # Multiplication retains a valid zero-gradient DDP connection.
            sampled=sampled*0
        return self.fuse(torch.cat((sampled.flatten(1,2),query),1))*roi[:,None].to(dtype)

    def _project(self,values,roi,local,actor,valid,pose):
        return self._project_impl(
            values,roi,local,actor,valid,pose,appearance_only=False
        )

    def _project_appearance(self,values,roi,local,appearance,valid,pose):
        return self._project_impl(
            values,roi,local,appearance,valid,pose,appearance_only=True
        )

    def _depth_route(self,roi,depth):
        if depth is None:
            return roi
        if depth.shape != roi.shape:
            raise ValueError("player depth must have the same shape as roi")
        active=roi>1e-6
        logits=roi.clamp_min(1e-6).log()-self.depth_sharpness*depth
        logits=logits.masked_fill(~active,-1e4)
        union=roi.amax(2,keepdim=True)
        routing=logits.softmax(2)*union
        return torch.where(active.any(2,keepdim=True),routing,
                           torch.zeros_like(routing))

    def route(self,roi,depth=None):
        """Return per-instance routing while preserving total ROI union mass."""
        return self._depth_route(roi,depth)

    def forward(self,reference,roi,local,actor,valid,pose,depth=None,
                appearance_only=False):
        b,t,a,h,w=roi.shape
        assert (h,w)==(self.height,self.width)
        assert reference.shape==(b,a,4,128,256)
        assert local.shape==(b,t,a,h,w,2) and actor.shape==(b,t,a,self.actor_dim)
        assert pose.shape==(b,t,a,8) and valid.shape==(b,a,4)
        values=self.value(self.norm(reference)).permute(0,1,2,4,3).reshape(
            b,a,4,self.value_channels,16,8
        )
        routing=self._depth_route(roi,depth)
        project=self._project_appearance if appearance_only else self._project
        active=(roi.detach().abs().flatten(-2).amax(-1)>0).nonzero(as_tuple=False)
        result=values.new_zeros((b*t,self.output_channels,h,w))
        for rows in active.split(8):
            if len(rows)==0:continue
            bi,ti,ai=rows.unbind(-1)
            args=(values[bi,ai],routing[bi,ti,ai],local[bi,ti,ai],actor[bi,ti,ai],valid[bi,ai],pose[bi,ti,ai])
            piece=checkpoint(project,*args,use_reentrant=False) if self.training and self.gradient_checkpointing and torch.is_grad_enabled() else project(*args)
            result=result.index_add(0,bi*t+ti,piece)
        # Empty-ROI batches must still be valid under DDP without unused params.
        zero=values.sum()*0+actor.sum()*0+sum(p.sum()*0 for p in self.fuse.parameters())
        return result.reshape(b,t,self.output_channels,h,w)+zero
