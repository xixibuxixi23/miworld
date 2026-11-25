import pytest
import torch

from miworld.training.renderer_trainer import (
    _sample_blockwise_train_time,
    _sample_unconditional_train_time,
)


def test_zero_reference_single_frame_has_a_trainable_noise_time():
    time = _sample_unconditional_train_time(
        batch_size=4,
        total_frames=1,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    assert time.shape == (4, 1)
    assert time.gt(0).all()
    assert time.lt(1).all()


def test_zero_reference_pure_noise_probability_one_forces_entire_clip():
    time = _sample_unconditional_train_time(
        batch_size=3,
        total_frames=1,
        device=torch.device("cpu"),
        ordinary_pure_noise_probability=1.0,
    )

    assert torch.equal(time, torch.ones(3, 1))


def test_ordinary_pure_noise_probability_one_forces_complete_future():
    time = _sample_blockwise_train_time(
        batch_size=7,
        total_frames=9,
        block_frames=4,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(1),
        ordinary_pure_noise_probability=1.0,
    )

    assert torch.equal(time[:, 0], torch.zeros(7))
    assert torch.equal(time[:, 1:], torch.ones(7, 8))


def test_ordinary_pure_noise_probability_is_sample_level_mixture():
    batch_size = 20_000
    time = _sample_blockwise_train_time(
        batch_size=batch_size,
        total_frames=9,
        block_frames=4,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(20260925),
        ordinary_pure_noise_probability=0.25,
    )

    pure_noise = time[:, 1:].eq(1).all(dim=1)
    observed = pure_noise.float().mean().item()
    assert 0.24 < observed < 0.26
    assert time[~pure_noise, 1:].lt(1).all()


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_ordinary_pure_noise_probability_must_be_valid(probability):
    with pytest.raises(ValueError, match="between 0 and 1"):
        _sample_blockwise_train_time(
            batch_size=1,
            total_frames=9,
            block_frames=8,
            device=torch.device("cpu"),
            ordinary_pure_noise_probability=probability,
        )
