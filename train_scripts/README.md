# Training entry points

| Model | Canonical entry point |
|---|---|
| M1 Fill | `train_m1.py` |
| M2 Transition/Write | `train_m2.py` |
| M3 Renderer | `train_m3_renderer.py` |
| M3 Policy | `train_m3_policy.py` |

The older entrypoints remain as checkpoint-compatible implementation modules.
New launches should use the four canonical wrappers above. M3 renderer and M3
policy are trained separately but loaded through one `miworld.M3` deployment
interface.

The current long-rollout M2 implementation is
`train_m2_long_rollout.py`. Cluster-neutral recipes for the selected M2 and M3
lineages live in `recipes/current/`; old per-machine launch scripts are kept
with their original run artifacts rather than on the main branch.
