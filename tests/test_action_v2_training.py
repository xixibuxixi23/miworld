from types import SimpleNamespace

import pytest
import torch

from miworld.models.action_chunk_decoder import ActionChunkDecoder
from miworld.models.structured_action import StructuredActionHead
from train_scripts.train_policy import (
    reduce_loss_parts,
    scheduled_learning_rate,
    shared_action_gradient_diagnostics,
)


def test_group_reduction_matches_objective_with_dense_mask_and_sample_weights():
    torch.manual_seed(91)
    head = StructuredActionHead(8, horizons=4, continuous_mouse=True)
    logits = head(torch.randn(6, 8))
    actions = torch.zeros(6, 4, 23)
    actions[..., 21:23] = torch.randn(6, 4, 2) * .02
    mask = torch.tensor([[[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]],
                         [[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 1]]], dtype=torch.bool)
    sample_weight = torch.tensor([.2, 2.])
    horizon_weight = torch.tensor([2., 2., 1., .5])
    loss, parts = head.loss(logits, actions, mask.flatten(0, 1), horizon_weight,
                            sample_weight.repeat_interleave(3), mouse_loss_weight=.1)
    reduced = reduce_loss_parts(parts, mask, horizon_weight, sample_weight)
    torch.testing.assert_close(loss, reduced['discrete_group'] + .1 * reduced['mouse_group'])
    legacy, _ = head.loss(logits, actions, mask.flatten(0, 1), horizon_weight,
                          sample_weight.repeat_interleave(3))
    torch.testing.assert_close(legacy, reduced['discrete_group'] + 2 * reduced['mouse_group'])


def test_cosine_schedule_boundaries_and_absolute_resume_steps():
    options = dict(peak_lr=1e-4, total_steps=100, warmup_steps=10,
                   schedule='cosine', min_lr_ratio=.01)
    assert scheduled_learning_rate(1, **options) == pytest.approx(1e-5)
    assert scheduled_learning_rate(10, **options) == pytest.approx(1e-4)
    assert scheduled_learning_rate(55, **options) == pytest.approx(5.05e-5)
    assert scheduled_learning_rate(100, **options) == pytest.approx(1e-6)
    uninterrupted = [scheduled_learning_rate(step, **options) for step in range(1, 101)]
    resumed = [scheduled_learning_rate(step, **options) for step in range(51, 101)]
    assert resumed == uninterrupted[50:]
    assert scheduled_learning_rate(1000, peak_lr=1e-5, total_steps=1000) == 1e-5


def test_shared_gradient_probe_does_not_mutate_training_gradients():
    torch.manual_seed(93)
    head = ActionChunkDecoder(16, 4, 8, decoder_hidden=16, layers=1, heads=4)
    torch.nn.init.normal_(head.output_projection.weight, std=.1)
    states = torch.randn(2, 3, 2, 16)
    valid = torch.ones(2, 3, dtype=torch.bool)
    text = torch.randn(2, 2, 8)
    text_mask = torch.ones(2, 2, dtype=torch.bool)
    logits = head(states, valid, text, text_mask, text, text_mask)
    loss, parts = head.loss(logits, torch.zeros(2, 4, 23), mouse_loss_weight=.1)
    reduced = reduce_loss_parts(parts, torch.ones(2, 4, dtype=torch.bool), torch.ones(4))
    raw = SimpleNamespace(families={'language_builder': SimpleNamespace(head=head)})
    record = shared_action_gradient_diagnostics(raw, reduced, .1)
    assert record['active'] and record['discrete_norm'] > 0 and record['mouse_raw_norm'] > 0
    assert record['mouse_weighted_norm'] == pytest.approx(.1 * record['mouse_raw_norm'])
    assert -1.00001 <= record['cosine'] <= 1.00001
    assert all(parameter.grad is None for parameter in head.parameters())
    loss.backward()
    assert head.decoder[0].self_attn.in_proj_weight.grad.abs().sum() > 0


def test_empty_mask_reduces_all_groups_to_zero():
    head = StructuredActionHead(8, horizons=4)
    _, parts = head.loss(head(torch.randn(2, 8)), torch.zeros(2, 4, 23))
    reduced = reduce_loss_parts(parts, torch.zeros(2, 4, dtype=torch.bool), torch.ones(4))
    assert all(value.item() == 0 for value in reduced.values())
