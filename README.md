# MIWORLD

This repository is the canonical source tree for the MIWORLD M1, M2, and M3
models. It intentionally contains code and tests only. Datasets, checkpoints,
training outputs, cluster environments, and one-off experiment snapshots are
kept in external artifact storage.

## Model layout

| Model | Responsibility | Canonical entry point |
| --- | --- | --- |
| M1 | Complete unobserved world voxels | `train_scripts/train_m1.py` |
| M2 | Predict motion, interactions, and world writes | `train_scripts/train_m2.py` |
| M3 renderer | Render first-person video | `train_scripts/train_m3_renderer.py` |
| M3 policy | Predict resident actions from video and task text | `train_scripts/train_m3_policy.py` |

The M3 renderer and policy use separate checkpoints but are exposed as one
combined model. The current policy source includes the future-latent spatial
action-conditioning path used by the 2026-11-25 `future_latent_action_v2`
lineage.

## Repository policy

- Production modules live under `miworld/`.
- Stable training entry points live under `train_scripts/`.
- Dataset construction utilities live under `dataset_toolkits/`.
- Reproducible behavior belongs in `tests/`.
- Cluster paths, GPU allocations, run directories, and credentials must never
  be committed.
- A new experiment should be expressed as a small config or recipe against the
  main branch, not as a copied source tree.

See [current experiment selections](docs/CURRENT_EXPERIMENTS.md) for the
checkpoint lineage that this source snapshot was built to load.

## Development

```bash
python -m pip install -e .
pytest -q
```

Large optional renderer dependencies and pretrained weights are installed by
the cluster environment rather than vendored into this repository.

