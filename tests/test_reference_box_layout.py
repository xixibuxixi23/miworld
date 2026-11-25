import math
import torch
from miworld.models.renderer_backbone.player_reference_box_layout import PlayerReferenceBoxLayout


def scene(x=0., distance=5., direction=(0.,1.,0.)):
    return dict(player_position=torch.tensor([[[[0.,0.,0.],[x,distance,0.]]]]),
        yaw_pitch=torch.zeros(1,1,2,2),player_valid=torch.ones(1,1,2,dtype=torch.bool),
        resident_type=torch.zeros(1,1,2,dtype=torch.long),
        camera_position=torch.tensor([[[[0.,0.,.9],[0.,0.,.9]]]]),
        camera_direction=torch.tensor(direction)[None,None,None].expand(1,1,2,3),
        camera=torch.full((1,1,2,1),math.pi/2))


def test_camera_player_excluded_and_far_box_not_full_screen():
    roi,views,local=PlayerReferenceBoxLayout()(scene(),torch.tensor([0]))
    assert roi.shape==(1,1,2,36,64) and local.shape==(1,1,2,36,64,2)
    assert roi[0,0,0].count_nonzero()==0 and views[0,0,0].count_nonzero()==0
    assert 0<float(roi[0,0,1].mean())<.2
    torch.testing.assert_close(views[0,0,1].sum(),torch.tensor(1.))


def test_optional_depth_matches_roi_and_excludes_target():
    roi,views,local,depth=PlayerReferenceBoxLayout()(
        scene(),torch.tensor([0]),return_depth=True)
    assert depth.shape==roi.shape and torch.isfinite(depth).all()
    assert depth[0,0,0].count_nonzero()==0
    assert (depth[roi>.15]>0).all()


def test_close_side_player_gets_low_confidence_full_frame_fallback():
    cond=scene(x=.65,distance=.35)
    cond['camera_position'][0,0,:,2]=1.6
    roi,_,_=PlayerReferenceBoxLayout()(cond,torch.tensor([0]))
    assert float(roi[0,0,1].min())>=.249


def test_box_grows_with_distance_and_preserves_offscreen_coordinates():
    layout=PlayerReferenceBoxLayout(); target=torch.tensor([0])
    near=layout(scene(distance=3),target)[0]
    far=layout(scene(distance=12),target)[0]
    assert near.sum()>far.sum()*3
    roi,_,local=layout(scene(x=3,distance=3),target)
    assert float(roi[0,0,1,:,63].max())>.9
    # The image edge is inside the original box, not its artificial right edge.
    assert abs(float(local[0,0,1,18,63,0]))<.8


def test_near_plane_intersection_and_behind_camera():
    layout=PlayerReferenceBoxLayout(); target=torch.tensor([0])
    for distance in (.01,.1,.5):
        result=layout(scene(distance=distance),target)
        assert all(torch.isfinite(v).all() for v in result)
        assert float(result[0][0,0,1].max())>.9
    roi,views,_=layout(scene(distance=-5),target)
    assert roi.count_nonzero()==0 and views.count_nonzero()==0


def test_invalid_resident_and_vertical_camera_are_finite():
    layout=PlayerReferenceBoxLayout(); target=torch.tensor([0])
    c=scene(direction=(0.,0.,1.))
    assert all(torch.isfinite(v).all() for v in layout(c,target))
    c=scene();c['player_valid'][0,0,1]=False
    roi,views,_=layout(c,target)
    assert roi.count_nonzero()==0 and views.count_nonzero()==0


def test_all_types_and_batches():
    c={k:v.expand(4,*v.shape[1:]).clone() for k,v in scene().items()}
    c['resident_type'][:,:,1]=torch.arange(4)[:,None]
    result=PlayerReferenceBoxLayout()(c,torch.zeros(4,dtype=torch.long))
    assert all(torch.isfinite(v).all() for v in result)
    assert (result[0][:,:,1].flatten(1).sum(-1)>0).all()
