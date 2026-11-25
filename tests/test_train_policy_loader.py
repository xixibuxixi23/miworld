import pytest
import torch
from types import SimpleNamespace

from train_scripts.train_policy import (
    loader_worker_options,
    policy_horizon_weights,
    should_snapshot,
)


def test_loader_worker_options_disable_hard_timeout_by_default():
    options = loader_worker_options(workers=1, timeout=0)

    assert options == {
        "timeout": 0,
        "persistent_workers": True,
        "multiprocessing_context": "spawn",
    }


def test_loader_worker_options_ignore_worker_only_settings_for_sync_loading():
    assert loader_worker_options(workers=0, timeout=0) == {}


@pytest.mark.parametrize("workers, timeout", [(-1, 0), (1, -1)])
def test_loader_worker_options_reject_negative_values(workers, timeout):
    with pytest.raises(ValueError):
        loader_worker_options(workers, timeout)


def test_execution_only_horizon_weights_leave_the_executed_prefix_active():
    model = SimpleNamespace(cfg=SimpleNamespace(horizons=32, execution_horizon=8))

    default = policy_horizon_weights(model, 'cpu')
    execution_only = policy_horizon_weights(model, 'cpu', tail_horizon_weight=0)

    torch.testing.assert_close(default[:8], torch.full((8,), 2.))
    torch.testing.assert_close(default[8:], torch.ones(24))
    torch.testing.assert_close(execution_only[:8], torch.full((8,), 2.))
    torch.testing.assert_close(execution_only[8:], torch.zeros(24))
    with pytest.raises(ValueError):
        policy_horizon_weights(model, 'cpu', tail_horizon_weight=-1)


def test_sixteen_horizon_weights_supervise_the_full_chunk():
    model = SimpleNamespace(cfg=SimpleNamespace(horizons=16, execution_horizon=8))
    weights = policy_horizon_weights(model, 'cpu', tail_horizon_weight=1)
    torch.testing.assert_close(weights[:8], torch.full((8,), 2.))
    torch.testing.assert_close(weights[8:], torch.ones(8))


def test_snapshot_schedule_can_retain_short_run_checkpoints():
    assert should_snapshot(250, 500, 250)
    assert should_snapshot(500, 500, 250)
    assert not should_snapshot(249, 500, 250)
    assert should_snapshot(499, 499, 1000)
    with pytest.raises(ValueError):
        should_snapshot(1, 2, 0)
