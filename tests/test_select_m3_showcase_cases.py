from dataset_toolkits.select_m3_showcase_cases import select_cases


def test_selects_one_early_prepare_case_per_episode_and_task():
    rows = []
    for task in ("wall", "floor"):
        for episode in range(3):
            rows.extend([
                {"showcase_task": task, "episode_id": f"{task}-{episode}", "anchor": 80,
                 "context_start": 48, "agent_slot": 0, "sampling_stratum": "background"},
                {"showcase_task": task, "episode_id": f"{task}-{episode}", "anchor": 64,
                 "context_start": 32, "agent_slot": 1, "sampling_stratum": "prepare"},
            ])
    selected = select_cases(rows, 2, 9)
    assert len(selected) == 4
    assert all(row["sampling_stratum"] == "prepare" for row in selected)
    assert all(row["anchor"] == 64 for row in selected)
    assert len({(row["showcase_task"], row["episode_id"]) for row in selected}) == 4
