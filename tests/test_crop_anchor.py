import numpy as np
import pytest
from miworld.pipelines.crop_anchor import CropAnchorTracker
from miworld.world_memory import WorldMemory


def test_stationary_half_block_noise_does_not_recenter():
    tracker=CropAnchorTracker()
    np.testing.assert_array_equal(tracker.update([[1,-3,6.500016]]),[[1,-3,7]])
    for z in [6.48154,6.4999,6.5001,6.457,6.6]:
        np.testing.assert_array_equal(tracker.update([[1,-3,z]]),[[1,-3,7]])


def test_walking_jumping_falling_and_teleport_recenter():
    tracker=CropAnchorTracker()
    tracker.update([[0,0,0]])
    for xyz in [[.8,0,0],[1.8,-1.8,1.8],[2.8,-2.8,0],[-20,30,-10]]:
        actual=tracker.update([xyz])
        np.testing.assert_array_equal(actual,np.floor(np.asarray([xyz])+.5))
        assert np.abs(np.asarray([xyz])-actual).max()<=.5


def test_hysteresis_prevents_boundary_chatter_and_allows_return():
    tracker=CropAnchorTracker();tracker.update([[0,0,0]])
    assert tracker.update([[.76,0,0]])[0,0]==1
    for x in [.49,.51,.3,.25]:assert tracker.update([[x,0,0]])[0,0]==1
    assert tracker.update([[.24,0,0]])[0,0]==0


def test_negative_coordinates_and_per_axis_per_resident_independence():
    tracker=CropAnchorTracker()
    tracker.update([[-1.4999,2,3.5001],[4,5,6]])
    np.testing.assert_array_equal(tracker.update([[-1.5001,2.8,3.49],[4,5,6]]),[[-1,3,4],[4,5,6]])


def test_zero_margin_exactly_matches_legacy_rounding():
    tracker=CropAnchorTracker(0)
    for p in [[[0,0,6.5]],[[.5,-.5,6.4999]],[[-10,4,7]]]:
        np.testing.assert_array_equal(tracker.update(p),np.floor(np.asarray(p)+.5))


def test_state_restore_copy_isolation_and_reset():
    tracker=CropAnchorTracker();returned=tracker.update([[0,0,6.5]]);returned[:]=999
    restored=CropAnchorTracker();restored.load_state_dict(tracker.state_dict())
    np.testing.assert_array_equal(restored.update([[0,0,6.48]]),[[0,0,7]])
    restored.reset()
    np.testing.assert_array_equal(restored.update([[0,0,6.48]]),[[0,0,6]])


def test_invalid_inputs_do_not_corrupt_state():
    tracker=CropAnchorTracker();tracker.update([[0,0,0]])
    for bad in [[[np.nan,0,0]],[[np.inf,0,0]],[[0,0]],[[0,0,0],[0,0,0]]]:
        with pytest.raises(ValueError):tracker.update(bad)
    np.testing.assert_array_equal(tracker.update([[0,0,0]]),[[0,0,0]])
    for margin in [-.1,.6,np.nan]:
        with pytest.raises(ValueError):CropAnchorTracker(margin)


def test_event_addresses_keep_same_global_target_after_recenter():
    tracker=CropAnchorTracker();memory=WorldMemory()
    target=np.array([1,-2,6]);memory.commit_points([target],[75])
    for p in [[1,-3,6.5],[1,-3,6.48],[1.9,-3,6.1]]:
        anchor=tracker.update([p])[0]
        local=target-(anchor-6)
        address=np.ravel_multi_index(tuple(local),(13,13,13))
        decoded=anchor-6+np.asarray(np.unravel_index(address,(13,13,13)))
        np.testing.assert_array_equal(decoded,target)
        values,known=memory.read_region(anchor-6,(13,13,13))
        assert known[tuple(local)] and values[tuple(local)]==75


def test_integer_world_translation_and_inputs_are_preserved():
    a,b=CropAnchorTracker(),CropAnchorTracker()
    shift=np.array([[-10,20,-5]])
    for p in [[[0,0,6.5001]],[[.6,-.6,6.4999]],[[1.8,-2,8]]]:
        p=np.asarray(p,dtype=np.float64);before=p.copy()
        np.testing.assert_array_equal(a.update(p)+shift,b.update(p+shift))
        np.testing.assert_array_equal(p,before)


