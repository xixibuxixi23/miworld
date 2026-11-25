# Current experiment selections

Verified on 2026-11-25. This file records model lineage and selection criteria;
it deliberately omits machine-specific paths.

## M2

- Lineage: `building_overfit_long8`
- Parent: engine-prior XYZ checkpoint at source step 5100
- Current run reached optimizer step 3250; latest complete saved checkpoint is
  step 3200.
- Recommended checkpoint: step 3100 (`best-position`), selected by the 64-frame
  position criterion.
- Model prior: horizontal engine motion, fixed 16-frame jump table, and a
  learned XYZ residual limited to 0.03 blocks per frame.
- Eight rollout blocks are trained; edit supervision includes the building
  subset and allows train/validation overlap for deliberate memorization.

## M3 policy

- Lineage: `future_latent_action_v2`
- Parent: `future_latent_v1` step 13800
- Training range: 13800 to 18800 on eight GPUs, weights-only migration into a
  new AdamW optimizer.
- Best validation loss: step 15000.
- Best place F1: step 18500 (F1 0.630531).
- Final step 18800 regressed slightly and is not the automatic evaluation
  default.
- The v2 source differs from v1 in exactly two implementation files and one
  new test: the predicted +8/+16 visual latent plan is spatially cross-attended
  by the action queries with a zero-initialized output projection.

## Selection rule

Always record the tested checkpoint separately from the latest saved checkpoint
and the live optimizer step. Checkpoints are distributed as immutable artifacts
with hashes; they are never committed to Git.

