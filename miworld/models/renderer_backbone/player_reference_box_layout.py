"""Opt-in oriented 3D-box routing, with near-plane clipping.

Dimensions are coarse routing priors, not measured collision/mesh bounds.
No GT mask, RGB or inferred image bbox enters this module. Legacy layout stays
available so existing runs/checkpoints do not silently change behavior.
"""
import itertools
import torch
from torch import nn
import torch.nn.functional as F


class PlayerReferenceBoxLayout(nn.Module):
    def __init__(self, height=36, width=64, near=.05, edge_sharpness=1.5,
                 core_horizontal_scale=.65, halo_weight=.25,
                 near_fallback_weight=.25, near_fallback_distance=2.25):
        super().__init__()
        self.height, self.width, self.near = height, width, near
        self.edge_sharpness = edge_sharpness
        self.core_horizontal_scale = float(core_horizontal_scale)
        self.halo_weight = float(halo_weight)
        self.near_fallback_weight = float(near_fallback_weight)
        self.near_fallback_distance = float(near_fallback_distance)
        # Width, front/back depth, height: human, villager, zombie, skeleton.
        # Include modest room for limbs; actual articulated bounds are unavailable.
        self.register_buffer('dimensions', torch.tensor([
            [1.10, .90, 1.90], [1.10, .95, 2.00],
            [1.10, 1.20, 1.95], [.95, 1.10, 2.05],
        ]), persistent=False)
        corners = list(itertools.product((-1., 1.), (-1., 1.), (0., 1.)))
        edges = [(i,j) for i in range(8) for j in range(i+1,8)
                 if sum(a!=b for a,b in zip(corners[i],corners[j]))==1]
        self.register_buffer('corners',torch.tensor(corners),persistent=False)
        self.register_buffer('edges',torch.tensor(edges),persistent=False)

    @staticmethod
    def _ray_box(origin, direction, lower, upper, minimum_t, forward_cosine):
        """Intersect forward camera rays with an oriented box in box space."""
        origin = origin[..., None, None, :]
        lower = lower[..., None, None, :]
        upper = upper[..., None, None, :]
        parallel = direction.abs() < 1e-6
        safe_direction = torch.where(parallel, torch.ones_like(direction), direction)
        first = (lower - origin) / safe_direction
        last = (upper - origin) / safe_direction
        axis_near, axis_far = torch.minimum(first, last), torch.maximum(first, last)
        inside = (origin >= lower) & (origin <= upper)
        inf = torch.full_like(axis_near, float('inf'))
        axis_near = torch.where(parallel, torch.where(inside, -inf, inf), axis_near)
        axis_far = torch.where(parallel, torch.where(inside, inf, -inf), axis_far)
        enter = axis_near.amax(-1)
        leave = axis_far.amin(-1)
        enter = torch.maximum(enter, minimum_t[:, :, None])
        hit = leave >= enter
        # Return view-space depth so different screen rays remain comparable.
        depth = enter * forward_cosine[:, :, None]
        return hit, depth

    @staticmethod
    def _soften_core(core, guard):
        shape = core.shape
        blurred = F.avg_pool2d(
            core.flatten(0, 2).unsqueeze(1), 3, stride=1, padding=1
        ).squeeze(1).reshape(shape)
        # Do not let antialiasing grow beyond the conservative guard box.
        return torch.maximum(core, blurred * .5) * guard

    def forward(self, cond, target, return_depth=False):
        position=cond['player_position'].float(); yaw=cond['yaw_pitch'][...,0].float()
        b,t,a,_=position.shape; idx=torch.arange(b,device=position.device)
        camera=cond['camera_position'][idx,:,target].float()
        forward=F.normalize(cond['camera_direction'][idx,:,target].float(),dim=-1,eps=1e-6)
        down_axis=forward.new_tensor([0,0,-1]).expand_as(forward)
        fallback=forward.new_tensor([0,-1,0]).expand_as(forward)
        down_axis=torch.where((forward[...,2].abs()>.999)[...,None],fallback,down_axis)
        right=F.normalize(torch.cross(down_axis,forward,dim=-1),dim=-1,eps=1e-6)
        down=F.normalize(torch.cross(forward,right,dim=-1),dim=-1,eps=1e-6)
        kind=cond['resident_type'].long()
        dims=self.dimensions[kind].to(position)
        core_scale=dims.new_tensor([self.core_horizontal_scale,
                                    self.core_horizontal_scale,1.])
        core_dims=dims*core_scale
        signs=self.corners.to(position)
        source_forward=torch.stack((yaw.sin(),yaw.cos(),torch.zeros_like(yaw)),-1)
        source_right=torch.stack((yaw.cos(),-yaw.sin(),torch.zeros_like(yaw)),-1)

        # Rasterize the oriented player box with camera rays instead of taking
        # a screen-space rectangle around projected corners. This remains valid
        # when the camera is very close to, or inside, the box.
        grid_x=torch.linspace(-1,1,self.width,device=position.device,dtype=position.dtype)
        grid_y=torch.linspace(-1,1,self.height,device=position.device,dtype=position.dtype)
        tan_x=torch.tan(cond['camera'][idx,:,target,-1].float().clamp(.05,3)/2)
        tan_y=tan_x*self.height/self.width
        rays=(forward[:,:,None,None,:]
              +right[:,:,None,None,:]*grid_x[None,None,None,:,None]*tan_x[:,:,None,None,None]
              +down[:,:,None,None,:]*grid_y[None,None,:,None,None]*tan_y[:,:,None,None,None])
        rays=F.normalize(rays,dim=-1,eps=1e-6)
        ray_forward=(rays*forward[:,:,None,None,:]).sum(-1).clamp_min(1e-6)
        minimum_t=self.near/ray_forward
        camera_delta=camera[:,:,None,:]-position
        origin=torch.stack(((camera_delta*source_right).sum(-1),
                            (camera_delta*source_forward).sum(-1),
                            camera_delta[...,2]),-1)
        direction=torch.stack(((rays[:,:,None]*source_right[:,:,:,None,None]).sum(-1),
                               (rays[:,:,None]*source_forward[:,:,:,None,None]).sum(-1),
                               rays[:,:,None,...,2].expand(-1,-1,a,-1,-1)),-1)

        def intersect(box_dims):
            lower=torch.stack((-box_dims[...,0]/2,-box_dims[...,1]/2,
                               box_dims[...,2]*0-.05),-1)
            upper=torch.stack((box_dims[...,0]/2,box_dims[...,1]/2,
                               box_dims[...,2]-.05),-1)
            return self._ray_box(origin,direction,lower,upper,minimum_t,ray_forward)

        core_hit,_=intersect(core_dims)
        guard_hit,guard_depth=intersect(dims)
        core=core_hit.to(position.dtype); guard=guard_hit.to(position.dtype)
        roi=torch.maximum(self._soften_core(core,guard),guard*self.halo_weight)

        # Keep the existing local-coordinate/reference sampling convention.
        # Only the routing mask changes; checkpoints retain compatible features.
        xyz=position[...,None,:]+source_right[...,None,:]*(dims[...,None,0:1]/2*signs[:,0:1])
        xyz=xyz+source_forward[...,None,:]*(dims[...,None,1:2]/2*signs[:,1:2])
        xyz=xyz+position.new_tensor([0,0,1])*(dims[...,None,2:3]*signs[:,2:3]-.05)
        relative=xyz-camera[:,:,None,None,:]
        points=torch.stack(((relative*right[:,:,None,None,:]).sum(-1),
            (relative*down[:,:,None,None,:]).sum(-1),
            (relative*forward[:,:,None,None,:]).sum(-1)),-1)
        first,last=points[...,self.edges[:,0],:],points[...,self.edges[:,1],:]
        dz=last[...,2]-first[...,2]
        crossing=((first[...,2]<self.near)&(last[...,2]>=self.near))|((last[...,2]<self.near)&(first[...,2]>=self.near))
        fraction=(self.near-first[...,2])/torch.where(dz.abs()>1e-8,dz,torch.ones_like(dz))
        intersection=first+fraction.clamp(0,1)[...,None]*(last-first)
        candidates=torch.cat((points,intersection),-2)
        usable=torch.cat((points[...,2]>=self.near,crossing),-1)
        tan_x=tan_x[:,:,None,None]
        depth=candidates[...,2].clamp_min(self.near)
        u=(candidates[...,0]/depth/tan_x+1)*(self.width-1)/2
        v=(candidates[...,1]/depth/(tan_x*self.height/self.width)+1)*(self.height-1)/2
        has_points=usable.any(-1)
        def bounds(values):
            lo=values.masked_fill(~usable,float('inf')).amin(-1)
            hi=values.masked_fill(~usable,float('-inf')).amax(-1)
            lo=torch.where(has_points,lo,torch.zeros_like(lo))
            hi=torch.where(has_points,hi,torch.zeros_like(hi))
            center=(hi+lo)/2
            # Keep original off-screen extents. Never cap the box at image size.
            radius=(hi-lo).clamp_min(1)/2+.5
            return center,radius
        cx,rx=bounds(u); cy,ry=bounds(v)
        x=torch.arange(self.width,device=position.device)[None,None,None,None,:]
        y=torch.arange(self.height,device=position.device)[None,None,None,:,None]
        dx=x-cx[...,None,None]; dy=y-cy[...,None,None]
        rectangle=torch.sigmoid((rx[...,None,None]-dx.abs())*self.edge_sharpness)
        rectangle=rectangle*torch.sigmoid(
            (ry[...,None,None]-dy.abs())*self.edge_sharpness
        )
        center_relative=position-camera[:,:,None]
        center_forward=(center_relative*forward[:,:,None]).sum(-1)
        center_distance=center_relative.norm(dim=-1)
        near_side=(center_distance<self.near_fallback_distance)&(center_forward<1.)
        near_side&=center_forward>-dims[...,1]
        near_crossing=crossing.any(-1)
        # A coarse box cannot describe articulated meshes that graze the side of
        # the near plane. Keep a low-confidence fallback rather than dropping a
        # close player's limbs or body entirely. It stays below the core weight.
        fallback=(near_side[...,None,None].to(roi.dtype)
                  +(near_crossing&~near_side)[...,None,None].to(roi.dtype)*rectangle
                 ).clamp_max(1)
        roi=torch.maximum(roi,fallback*self.near_fallback_weight)
        valid=cond['player_valid'].bool()
        visible=(guard_hit.flatten(-2).any(-1)|near_side|near_crossing)
        visible&=valid & valid[idx,:,target][...,None]
        visible &= torch.arange(a,device=position.device)[None,None,:]!=target[:,None,None]
        roi*=visible[...,None,None]
        local=torch.stack(((dx/rx[...,None,None]).expand(-1,-1,-1,self.height,-1),
            (dy/ry[...,None,None]).expand(-1,-1,-1,-1,self.width)),-1).clamp(-2,2)
        to_camera=F.normalize(camera[:,:,None,:2]-position[...,:2],dim=-1,eps=1e-6)
        front=(to_camera*source_forward[...,:2]).sum(-1)
        side=(to_camera*source_right[...,:2]).sum(-1)
        views=(torch.stack((front,-front,-side,side),-1)*8).softmax(-1)*visible[...,None]
        result=(roi.contiguous(),views.contiguous(),local.contiguous())
        if not return_depth:
            return result
        fallback_pixels=fallback>1e-6
        fallback_depth=center_forward.clamp_min(self.near)
        routed_depth=torch.where(fallback_pixels,fallback_depth[...,None,None],
                                 guard_depth)
        routed_hit=guard_hit|fallback_pixels
        depth=torch.where(routed_hit & visible[...,None,None],routed_depth,
                          torch.zeros_like(routed_depth))
        return (*result,depth.contiguous())
