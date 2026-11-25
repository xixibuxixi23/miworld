"""Training infrastructure contracts, without starting a training job."""
import json

import pytest
import torch

from miworld.models.local_world_writer_v2 import LocalWorldWriterV2, LocalWorldWriterV2Args
from miworld.training.m2_v2_rollout import supervised_velocity_loss
from train_scripts.train_m2_tristream_v2 import CachedRollouts, edit_metrics, parameter_groups, rate_metrics


def cached_episode(root, name, starts, revision=4):
    directory = root / "train"
    directory.mkdir(parents=True, exist_ok=True)
    samples = []
    for start in starts:
        samples.append({
            "metadata": {"start": start},
            "inputs": {"active": torch.ones(2, dtype=torch.bool),
                       "actions": torch.full((8, 2, 23), float(start))},
            "targets": {"block_valid": torch.zeros(8, 2, dtype=torch.bool),
                        "address_valid": torch.zeros(8, 2, dtype=torch.bool),
                        "address": torch.zeros(8, 2, dtype=torch.long)},
        })
    torch.save({"transition_cache_revision": revision, "timesteps": 8,
                "samples": samples}, directory / (name + ".pt"))


def index_file(root, names):
    index = root / "index.json"
    index.write_text(json.dumps({"splits": {"train": [{"path": name} for name in names]}}))
    return index


def test_cached_catalog_is_read_only_and_uses_only_authorized_split(tmp_path):
    index = index_file(tmp_path, ["present", "not_prebuilt"])
    cached_episode(tmp_path, "present", range(0, 64, 8))
    cached_episode(tmp_path, "unindexed", range(0, 64, 8))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.pt")}
    data = CachedRollouts(index, tmp_path, "train", 4)
    assert len(data.entries) == 2
    assert {entry["episode"] for entry in data.entries} == {"present"}
    assert data.discovery["missing_index_caches"] == 1
    assert {p: p.read_bytes() for p in tmp_path.rglob("*.pt")} == before


def test_invalid_revision_fails_without_overwriting_cache(tmp_path):
    index = index_file(tmp_path, ["old"])
    cached_episode(tmp_path, "old", range(0, 64, 8), revision=3)
    cache = tmp_path / "train/old.pt"
    before = cache.read_bytes()
    with pytest.raises(ValueError, match="revision/horizon mismatch"):
        CachedRollouts(index, tmp_path, "train", 4)
    assert cache.read_bytes() == before


def test_no_complete_sequence_fails_instead_of_looping(tmp_path):
    index = index_file(tmp_path, ["short"])
    cached_episode(tmp_path, "short", [0, 8, 32, 40])
    with pytest.raises(ValueError, match="no complete"):
        CachedRollouts(index, tmp_path, "train", 4)


def test_sampling_is_reproducible_at_resume_step_and_rank(tmp_path):
    index = index_file(tmp_path, ["episode"])
    cached_episode(tmp_path, "episode", range(0, 160, 8))
    first = CachedRollouts(index, tmp_path, "train", 4)
    second = CachedRollouts(index, tmp_path, "train", 4)
    batches_a, indices_a = first.training_batch(103, 2, 3, 7, 4)
    # Unrelated earlier calls cannot change resumed sampling.
    second.training_batch(1, 0, 2, 7, 1)
    batches_b, indices_b = second.training_batch(103, 2, 3, 7, 4)
    assert first.signature == second.signature
    assert indices_a == indices_b
    for a, b in zip(batches_a, batches_b):
        torch.testing.assert_close(a["inputs"]["actions"], b["inputs"]["actions"])


def test_broadcast_catalog_needs_no_scan_or_cache_read(tmp_path):
    index = index_file(tmp_path, ["episode"])
    cached_episode(tmp_path, "episode", range(0, 64, 8))
    original = CachedRollouts(index, tmp_path, "train", 4)
    index.unlink()
    (tmp_path / "train/episode.pt").unlink()
    replicated = CachedRollouts(index, tmp_path, "train", 4, catalog=original.catalog())
    assert replicated.signature == original.signature
    assert replicated.entries == original.entries


def test_edit_precision_penalizes_spurious_edits_but_not_unscorable_frames():
    # One correct edit, one spurious edit on a null-GT frame, one out-of-memory
    # GT edit. The latter is excluded and its coverage loss remains visible.
    inputs = {"active": torch.ones(1, 1, dtype=torch.bool)}
    targets = {"address": torch.tensor([[[1], [2198], [-1]]]),
               "address_valid": torch.tensor([[[True], [True], [False]]]),
               "state_valid": torch.ones(1, 3, 1, dtype=torch.bool),
               "block_valid": torch.tensor([[[True], [False], [True]]]),
               "_original_edit_valid": torch.tensor([[[True], [False], [True]]]),
               "_fixed_edit_covered": torch.tensor([[[True], [False], [False]]]),
               "_fixed_edit_uncovered": torch.tensor([[[False], [False], [True]]])}
    decoded = {"address": torch.tensor([[[1], [2], [3]]])}
    carry = {"_fixed_shared_write_count": torch.tensor(3), "_fixed_same_frame_conflicts": torch.tensor(0)}
    metrics = edit_metrics(inputs, targets, decoded, carry)
    metrics.update(correct_edits=1, block_labels=1)
    result = rate_metrics(metrics)
    assert result["predicted_edits"] == 2
    assert result["edit_precision"] == .5
    assert result["edit_recall"] == 1
    assert result["edit_f1"] == pytest.approx(2 / 3)
    assert result["edit_memory_coverage"] == .5


def test_optimizer_groups_cover_all_parameters_exactly_once():
    model = LocalWorldWriterV2(LocalWorldWriterV2Args(
        num_block_classes=3, num_items=4, air_class=0,
        width=16, depth=1, heads=2, event_decoder_layers=1,
    ))
    groups, assignments = parameter_groups(model, 3e-5, 1e-4, .01)
    params = [parameter for group in groups for parameter in group["params"]]
    assert len(params) == len({id(parameter) for parameter in params})
    assert {id(parameter) for parameter in params} == {id(parameter) for parameter in model.parameters()}
    assert assignments["state_residual_head.weight"] == "new"
    assert assignments["blocks.0.action_attention.in_proj_weight"] == "new"
    assert assignments["blocks.0.temporal_attention.in_proj_weight"] == "reused"
    assert assignments["memory_encoder.block_embedding.weight"] == "reused"


def test_supervised_velocity_loss_uses_initial_pose_then_consecutive_targets():
    initial = torch.zeros(1, 1, 5)
    pose = torch.zeros(1, 3, 1, 5)
    pose[0, :, 0, 0] = torch.tensor([0.1, 0.3, 0.6])
    expected = torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1)
    output = {"velocity": torch.cat((expected, torch.zeros(1, 3, 1, 2)), dim=-1)}
    inputs = {"initial_pose": initial, "active": torch.ones(1, 1, dtype=torch.bool)}
    targets = {"pose": pose, "state_valid": torch.ones(1, 3, 1, dtype=torch.bool)}
    loss, metrics = supervised_velocity_loss(output, inputs, targets)
    assert float(loss) == pytest.approx(0., abs=1e-7)
    assert metrics["velocity_error"] == pytest.approx(0., abs=1e-7)
    output["velocity"][0, 1, 0, 0] += 1
    loss, metrics = supervised_velocity_loss(output, inputs, targets)
    assert loss > 0
    assert metrics["velocity_error"] > 0


def test_conditioned_velocity_migration_preserves_inherited_tensors_and_resets_new_semantics():
    from dataclasses import asdict
    from diagnostics.migrate_m2_v2_conditioned_integrated_velocity import migrate

    source_model = LocalWorldWriterV2(LocalWorldWriterV2Args(
        num_block_classes=3, num_items=4, air_class=0,
        width=16, depth=1, heads=2, event_decoder_layers=1,
        position_prior="observed_velocity",
    ))
    with torch.no_grad():
        source_model.state_residual_head.weight.fill_(0.7)
        source_model.state_residual_head.bias.fill_(0.4)
    source = {"model": source_model.state_dict(),
              "config": {"schema": "m2-local-world-writer-v2", "model": asdict(source_model.cfg)},
              "step": 123}
    state, config, metadata = migrate(source)
    assert config["model"]["velocity_mode"] == "conditioned_integrated"
    assert config["model"]["position_prior"] == "action"
    torch.testing.assert_close(state["held_head.weight"], source["model"]["held_head.weight"])
    assert torch.count_nonzero(state["state_residual_head.weight"][:3]) == 0
    assert torch.count_nonzero(state["state_residual_head.weight"][3:]) > 0
    assert torch.count_nonzero(state["velocity_encoder.weight"]) == 0
    assert metadata["source_step"] == 123
