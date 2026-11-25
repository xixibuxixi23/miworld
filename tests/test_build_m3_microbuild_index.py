"""Contract checks runnable without Torch or NumPy on the local laptop."""
import json
from pathlib import Path
import tempfile
import unittest

from dataset_toolkits.build_m3_microbuild_index import (
    assign_splits, discover, identity_keys, verify_fingerprints, window_specs,
)


class MicrobuildIndexTests(unittest.TestCase):
    def test_early_active_episode_uses_anchor_32_not_128(self):
        specs, audit = window_specs(frames=301, model_start=0,
                                   active_start=40, active_end_inclusive=70)
        self.assertEqual(specs, [{"context_start": 8, "anchor": 40,
                                 "absolute_supervision_ranges": [[40, 79]]}])
        self.assertEqual(audit["early_active_steps_unavailable"], 0)

    def test_active_ranges_are_complete_disjoint_and_limit_wait(self):
        specs, _ = window_specs(frames=301, model_start=0,
                                active_start=40, active_end_inclusive=205)
        observed = []
        for spec in specs:
            start, end = spec["absolute_supervision_ranges"][0]
            self.assertGreaterEqual(start, spec["context_start"] + 32)
            self.assertLessEqual(end, spec["context_start"] + 104)
            self.assertLessEqual(spec["context_start"] + 129, 301)
            observed.extend(range(start, end))
        self.assertEqual(observed, list(range(40, 214)))
        self.assertEqual(len(observed), len(set(observed)))

    def test_unavailable_early_actions_are_explicit_and_not_padded(self):
        _, audit = window_specs(frames=301, model_start=0,
                                active_start=5, active_end_inclusive=50)
        self.assertEqual(audit["early_active_steps_unavailable"], 27)
        with self.assertRaisesRegex(ValueError, "all active actions"):
            window_specs(frames=301, model_start=0, active_start=0, active_end_inclusive=20)
        with self.assertRaisesRegex(ValueError, "active tail"):
            window_specs(frames=301, model_start=0, active_start=250, active_end_inclusive=299)

    def test_shared_world_or_seed_never_cross_split(self):
        episodes = [
            {"identity_keys": identity_keys({"seed": 1, "world_id": "a"})},
            {"identity_keys": identity_keys({"seed": 2, "world_id": "a"})},
            {"identity_keys": identity_keys({"seed": 2, "world_id": "b"})},
            {"identity_keys": identity_keys({"seed": 3, "world_id": "other"})},
        ]
        assign_splits(episodes, seed=7, dev_fraction=.15, test_fraction=.15)
        self.assertEqual(len({episode["split_group"] for episode in episodes[:3]}), 1)
        self.assertEqual(len({episode["split"] for episode in episodes[:3]}), 1)
        reversed_rows = [{"identity_keys": episode["identity_keys"]} for episode in reversed(episodes)]
        assign_splits(reversed_rows, seed=7, dev_fraction=.15, test_fraction=.15)
        self.assertEqual([episode["split"] for episode in episodes],
                         [episode["split"] for episode in reversed(reversed_rows)])

    def test_snapshot_filters_rejections_and_detects_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, usable in enumerate((True, False)):
                episode = root / f"episode{index}"
                episode.mkdir()
                (episode / "simple_build_pilot.json").write_text(json.dumps(
                    {"main_set_usable": usable, "structure": "wall3", "task_id": "wall"}))
                (episode / "manifest.json").write_text(json.dumps(
                    {"seed": index, "task_id": "wall"}))
                (episode / "data.npz").write_bytes(b"test fingerprint payload")
            episodes, rejected = discover(root)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(rejected, {"main_set_unusable": 1})
            verify_fingerprints(episodes)
            (root / "episode0" / "data.npz").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "source changed"):
                verify_fingerprints(episodes)


if __name__ == "__main__":
    unittest.main()
