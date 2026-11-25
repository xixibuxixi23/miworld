import json

from dataset_toolkits.build_m3_simple_atomic_index import (
    classify_task,
    select_rows,
    split_episodes,
)


def test_classify_task_keeps_small_atomic_and_rejects_composite():
    assert classify_task("build_wall_4x2x1_cobble") == ("wall", (4, 2, 1), "cobble")
    assert classify_task("build_cuboid_3x2x2_wood") == ("cuboid", (3, 2, 2), "wood")
    assert classify_task("dig_hole_3x1x4") is None
    assert classify_task("build_wall_6x3x1_cobble") is None
    assert classify_task("episode_build_house_wood") is None
    assert classify_task("dig_then_fill_3x2x2_layers_dirt") is None


def test_select_rows_preserves_labels_and_text(tmp_path):
    source = tmp_path / "source.jsonl"
    kept = {
        "episode_id": "episode-a",
        "profile": "language_builder",
        "task_id": "build_floor_3x1x3_glass",
        "task_text": "Lay a 3 by 3 floor using glass.",
        "sampling_stratum": "event",
        "first8_action_counts": {"place": 2, "dig": 0, "move": 1},
    }
    removed = {**kept, "episode_id": "episode-b", "task_id": "episode_tower_bridge_wood"}
    source.write_text("\n".join(json.dumps(row) for row in (kept, removed)) + "\n")
    rows, excluded = select_rows(source)
    assert len(rows) == 1
    assert rows[0]["task_text"] == kept["task_text"]
    assert rows[0]["first8_action_counts"] == kept["first8_action_counts"]
    assert rows[0]["simple_task_family"] == "floor"
    assert excluded["non_atomic_or_large_task"] == 1


def test_split_is_episode_disjoint():
    rows = []
    for family in ("wall", "floor"):
        for episode in range(10):
            rows.append({"simple_task_family": family, "episode_id": f"{family}-{episode}"})
    train, dev = split_episodes(rows, 0.2, 7)
    assert {row["episode_id"] for row in train}.isdisjoint(
        {row["episode_id"] for row in dev}
    )
    assert len(dev) == 4
