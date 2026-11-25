import math

from dataset_toolkits.build_m3_showcase_index import (
    DEFAULT_TASK_PREFIXES,
    balanced_dev,
    filter_rows,
    set_weights,
    split_episodes,
)


def rows():
    result = []
    specs = {
        "floor_3x1x2": "build_floor_3x1x2_stonebrick",
        "wall_4x2x1": "build_wall_4x2x1_stonebrick",
        "cuboid_3x1x2": "build_cuboid_3x1x2_stonebrick",
    }
    for task, task_id in specs.items():
        for episode in range(5):
            for stratum in ("prepare", "event", "background"):
                for clip in range(2):
                    result.append({
                        "profile": "language_builder",
                        "task_id": task_id,
                        "episode_id": f"{task}-{episode}",
                        "sampling_stratum": stratum,
                        "first8_action_counts": {
                            "n": 72,
                            "move": 8 if stratum == "prepare" else 0,
                            "place": 1 if stratum == "event" else 0,
                        },
                        "clip": clip,
                    })
    result.append({"profile": "zombie_melee", "task_id": "build_wall_4x2x1_stonebrick"})
    return result


def test_filter_split_weights_and_balanced_dev():
    selected = filter_rows(rows(), DEFAULT_TASK_PREFIXES, {"stonebrick"})
    assert len(selected) == 3 * 5 * 3 * 2
    train, dev_pool = split_episodes(selected, .2, 7)
    assert not ({r["episode_id"] for r in train} & {r["episode_id"] for r in dev_pool})
    assert {r["showcase_task"] for r in train} == set(DEFAULT_TASK_PREFIXES)
    set_weights(train)
    assert math.isclose(sum(r["sample_weight"] for r in train), 1.0)
    by_task = {}
    for task in DEFAULT_TASK_PREFIXES:
        by_task[task] = sum(r["sample_weight"] for r in train if r["showcase_task"] == task)
    assert math.isclose(by_task["wall_4x2x1"], .5)
    assert math.isclose(by_task["floor_3x1x2"], .3)
    assert math.isclose(by_task["cuboid_3x1x2"], .2)
    dev = balanced_dev(dev_pool, 2, 7)
    assert len(dev) == 3 * 3 * 2
    assert all(r["evaluation_label"] == "showcase_dev_episode_holdout" for r in dev)


def test_material_filter_and_profile_filter():
    source = rows()
    source.append({
        "profile": "language_builder",
        "task_id": "build_wall_4x2x1_glass",
        "episode_id": "glass",
        "sampling_stratum": "event",
    })
    selected = filter_rows(source, DEFAULT_TASK_PREFIXES, {"glass"})
    assert len(selected) == 1
    assert selected[0]["episode_id"] == "glass"
