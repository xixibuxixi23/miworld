# Production models

The public names are `M1`, `M2`, and `M3`.

- M1 is the latest unified flow fill model in `m1_flow.py`.
- M2 is the Local World Writer in `local_world_writer.py`.
- M3 combines the renderer in `renderer.py` with the inserted policy in
  `inserted_policy.py`; `m3.py` provides the unified loading interface.

Older fill and transition implementations remain importable for checkpoint and
experiment compatibility, but they are not the default public aliases.
