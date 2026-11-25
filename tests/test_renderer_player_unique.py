import pytest
import torch

from miworld.training.renderer_diagnostics import pack_player_diagnostics, summarize_player_diagnostics
from miworld.training.renderer_trainer import select_renderer_pixel_frames, renderer_pixel_losses, renderer_training_losses
from tests.test_renderer_player_pixels_only import ToyRenderer, FrozenCodec


def select(mask, count=2, seed=19):
    return select_renderer_pixel_frames(
        torch.ones_like(mask), frames_per_sample=count, player_region_mask=mask,
        selection_mode="player_unique", generator=torch.Generator().manual_seed(seed),
    )[0]


def test_unique_player_frames_are_visible_distinct_and_exclude_prefix():
    mask = torch.zeros(64, 9, 1, 8, 8)
    mask[:, 0] = 1
    mask[:, 2, :, :2] = 1
    mask[:, 7] = 1
    indices = select(mask)
    assert torch.equal(indices.sort(1).values, torch.tensor([[2, 7]]).expand(64, -1))
    assert torch.equal(indices, select(mask))


def test_one_visible_frame_is_not_duplicated_and_empty_clips_do_not_crash():
    mask = torch.zeros(64, 9, 1, 8, 8)
    mask[:, 5] = 1
    indices = select(mask)
    assert (indices[:, 0] == 5).all()
    assert (indices[:, 1] != 5).all()
    assert ((indices > 0) & (indices < 9)).all()
    empty = select(torch.zeros_like(mask), count=8)
    assert torch.equal(empty.sort(1).values, torch.arange(1, 9).expand(64, -1))
    with pytest.raises(ValueError, match="requires player_region_mask"):
        select_renderer_pixel_frames(mask, frames_per_sample=2, selection_mode="player_unique")


def test_stratified_player_frames_cover_time_and_remain_distinct():
    mask = torch.zeros(16, 9, 1, 8, 8)
    mask[:, [2, 4, 6, 8]] = 1
    indices = select_renderer_pixel_frames(
        torch.ones_like(mask), frames_per_sample=4, player_region_mask=mask,
        selection_mode="player_stratified", generator=torch.Generator().manual_seed(11),
    )[0]
    assert torch.equal(indices, torch.tensor([[2, 4, 6, 8]]).expand(16, -1))

    sparse = torch.zeros_like(mask)
    sparse[:, [2, 8]] = 1
    indices = select_renderer_pixel_frames(
        torch.ones_like(sparse), frames_per_sample=4, player_region_mask=sparse,
        selection_mode="player_stratified", generator=torch.Generator().manual_seed(12),
    )[0]
    assert (indices.sort(1).values[:, 1:] != indices.sort(1).values[:, :-1]).all()
    assert ((indices == 2).any(1) & (indices == 8).any(1)).all()


def test_player_ring_l1_penalizes_spill_and_balances_frames():
    class Codec:
        def decode_for_loss(self, latent, chunk_size=1):
            return latent

    predicted = torch.zeros(1, 3, 3, 9, 9, requires_grad=True)
    target = torch.zeros_like(predicted)
    mask = torch.zeros(1, 3, 1, 9, 9)
    mask[:, 1, :, 4, 4] = 1
    mask[:, 2, :, 2:7, 2:7] = 1
    predicted.data[:, 1, :, 4, 4] = 1
    predicted.data[:, 2, :, 2:7, 2:7] = 0.5
    predicted.data[:, 1, :, 4, 5] = 1  # Error only in the outer ring.
    losses = renderer_pixel_losses(
        Codec(), predicted, target, mask, player_region_mask=mask,
        frames_per_sample=2, frame_selection="player_stratified",
        player_ring_radius=1, player_ring_weight=0.25,
        generator=torch.Generator().manual_seed(4),
    )
    # The extra neighboring error proves that the outside ring is supervised;
    # each frame is normalized independently before the two are averaged.
    assert 0.4 < losses["player_pixel_l1"] < 0.5
    losses["player_pixel_l1"].backward()
    assert predicted.grad[:, 1, :, 4, 5].abs().sum() > 0


def test_high_resolution_player_roi_loss_is_differentiable_and_ignores_empty_frames():
    class Codec:
        def decode_for_loss(self, latent, chunk_size=1):
            return latent

    predicted = torch.zeros(1, 3, 3, 32, 32, requires_grad=True)
    target = torch.zeros_like(predicted)
    mask = torch.zeros(1, 3, 1, 32, 32)
    mask[:, 1, :, 10:18, 14:18] = 1
    target.data[:, 1, :, 10:14, 14:18] = 1
    losses = renderer_pixel_losses(
        Codec(), predicted, target, torch.ones_like(mask),
        player_region_mask=mask, frames_per_sample=2,
        frame_selection="player_unique", player_roi_size=(128, 64),
        player_roi_padding=0.25, player_ring_radius=4,
        generator=torch.Generator().manual_seed(4),
    )
    assert losses["player_roi_l1"] > 0
    assert losses["player_roi_edge"] > 0
    (losses["player_roi_l1"] + losses["player_roi_edge"]).backward()
    assert predicted.grad[:, 1].abs().sum() > 0
    assert predicted.grad[:, 0].abs().sum() == 0
    assert predicted.grad[:, 2].abs().sum() == 0


def test_high_resolution_player_roi_weights_are_added_to_total_loss():
    torch.manual_seed(73)
    model, codec = ToyRenderer(), FrozenCodec()
    clean, rgb = torch.randn(2, 9, 3, 8, 8), torch.rand(2, 9, 3, 8, 8)
    mask = torch.ones(2, 9, 1, 8, 8)
    losses = renderer_training_losses(
        model, codec, clean, {}, rgb, mask, player_region_mask=mask,
        frames_per_sample=2, pixel_frame_selection="player_unique",
        flow_loss_weight=0., entity_pixel_l1_weight=0.,
        entity_pixel_edge_weight=0., health_pixel_l1_weight=0.,
        player_roi_l1_weight=.5, player_roi_edge_weight=.25,
        player_roi_size=(32, 16), generator=torch.Generator().manual_seed(17),
    )
    expected = .5 * losses["player_roi_l1"] + .25 * losses["player_roi_edge"]
    torch.testing.assert_close(losses["total_loss"], expected)


def test_pixel_noise_bins_follow_selected_frame_times_and_ignore_empty_masks():
    class Codec:
        def decode_for_loss(self, latent, chunk_size=1):
            return latent

    latent = torch.zeros(1, 9, 3, 8, 8)
    latent[:, 1] = .2
    latent[:, 3] = .8
    latent.requires_grad_()
    mask = torch.zeros(1, 9, 1, 8, 8)
    mask[:, [1, 3]] = 1
    times = torch.full((1, 8), .5)
    times[:, 0], times[:, 2] = .1, .9
    metrics = {}
    losses = renderer_pixel_losses(
        Codec(), latent, torch.zeros_like(latent), mask,
        player_region_mask=mask, frames_per_sample=2, frame_selection="player_unique",
        diagnostics=metrics, noise_time=times, generator=torch.Generator().manual_seed(6),
    )
    summary = summarize_player_diagnostics(pack_player_diagnostics(metrics, device="cpu"))
    assert summary["player_pixel_low_mean"] == pytest.approx(.2)
    assert summary["player_pixel_high_mean"] == pytest.approx(.8)
    assert summary["player_pixel_mid_count"] == 0
    assert "player_pixel_mid_mean" not in summary
    assert summary["player_pixel_supervised_fraction"] == 1
    assert not metrics["player_pixel_high_sum"].requires_grad
    losses["player_pixel_l1"].backward()
    assert latent.grad[:, 1].abs().sum() > 0
    assert latent.grad[:, 0].abs().sum() == 0


def test_noise_statistics_merge_sums_and_counts_not_rank_means():
    a = pack_player_diagnostics({"player_flow_low_sum": 2., "player_flow_low_count": 1,
                                "player_pixel_valid_count": 1, "player_pixel_selected_count": 2}, device="cpu")
    b = pack_player_diagnostics({"player_flow_low_sum": 18., "player_flow_low_count": 3,
                                "player_pixel_valid_count": 2, "player_pixel_selected_count": 2}, device="cpu")
    summary = summarize_player_diagnostics(a + b)
    assert summary["player_flow_low_mean"] == 5
    assert summary["player_flow_low_count"] == 4
    assert "player_flow_high_mean" not in summary
    assert summary["player_pixel_supervised_fraction"] == .75


def test_combined_diagnostics_do_not_change_loss_or_gradient():
    torch.manual_seed(73)
    model, codec = ToyRenderer(), FrozenCodec()
    clean, rgb = torch.randn(2, 9, 3, 8, 8), torch.rand(2, 9, 3, 8, 8)
    mask = torch.ones(2, 9, 1, 8, 8)
    results, gradients = [], []
    diagnostics = {}
    for collector in [None, diagnostics]:
        losses = renderer_training_losses(
            model, codec, clean, {}, rgb, mask, player_region_mask=mask,
            frames_per_sample=2, pixel_frame_selection="player_unique",
            flow_loss_weight=1., player_flow_loss_weight=1.,
            player_pixel_l1_weight=1., player_pixel_edge_weight=.25,
            entity_pixel_l1_weight=0., entity_pixel_edge_weight=0., health_pixel_l1_weight=0.,
            flow_diagnostics=collector, generator=torch.Generator().manual_seed(17),
        )
        expected = losses["flow_loss"] + losses["player_flow_loss"] + losses["player_pixel_l1"] + .25*losses["player_pixel_edge"]
        torch.testing.assert_close(losses["total_loss"], expected)
        results.append(losses["total_loss"].detach())
        gradients.append(torch.autograd.grad(losses["total_loss"], model.weight)[0])
    torch.testing.assert_close(*results)
    torch.testing.assert_close(*gradients)
    assert sum(int(diagnostics[f"player_flow_{k}_count"]) for k in ["low", "mid", "high"]) == 16
    assert sum(int(diagnostics[f"player_pixel_{k}_count"]) for k in ["low", "mid", "high"]) == 4


