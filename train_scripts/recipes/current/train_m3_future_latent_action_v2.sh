#!/usr/bin/env bash
set -euo pipefail

: "${MIWORLD_ROOT:?set MIWORLD_ROOT to this repository}"
: "${PYTHON_BIN:?set PYTHON_BIN to the training environment python}"
: "${DATASET_ROOT:?set the episode dataset root}"
: "${TRAIN_INDEX:?set the simple-atomic training index}"
: "${VAL_INDEX:?set the simple-atomic validation index}"
: "${BLOCK_VOCABULARY:?set the block vocabulary JSON}"
: "${TEXT_CATALOG:?set the text catalog JSON}"
: "${TEXT_CACHE:?set the encoded text cache}"
: "${M3_RENDERER_CHECKPOINT:?set the frozen M3 renderer checkpoint}"
: "${WAN_VAE_CHECKPOINT:?set the WAN VAE checkpoint}"
: "${M3_POLICY_PARENT:?set the future-latent parent checkpoint}"
: "${OUTPUT_DIR:?set a new output directory}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTHONPATH="$MIWORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"

exec "$PYTHON_BIN" -m torch.distributed.run --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  "$MIWORLD_ROOT/train_scripts/train_policy.py" \
  --dataset-root "$DATASET_ROOT" \
  --train-index "$TRAIN_INDEX" \
  --val-index "$VAL_INDEX" \
  --vocabulary "$BLOCK_VOCABULARY" \
  --text-catalog "$TEXT_CATALOG" \
  --text-cache "$TEXT_CACHE" \
  --m3-checkpoint "$M3_RENDERER_CHECKPOINT" \
  --wan-vae "$WAN_VAE_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --warm-start-future-latent "$M3_POLICY_PARENT" \
  --validate-on-resume \
  --policy-blocks 0 \
  --structured-inputs \
  --current-text-mode structured_goal_only \
  --current-text-scale 2 \
  --mask-component-boundaries \
  --builder-visual-queries 1 \
  --builder-spatial-tokens 8 \
  --builder-action-decoder-layers 12 \
  --builder-action-decoder-hidden 1024 \
  --builder-action-decoder-heads 16 \
  --builder-action-architecture joint_v2 \
  --builder-action-visual-grid 8 \
  --builder-npc-attention \
  --builder-text-only-visual-queries \
  --builder-action-history 64 \
  --builder-interaction-pairs \
  --builder-history-action-dropout .25 \
  --builder-text-action-film \
  --builder-text-action-film-hidden 256 \
  --builder-future-latent-channels 16 \
  --builder-future-latent-grid 8 \
  --future-latent-loss-weight .05 \
  --horizons 16 \
  --dense-stride 8 \
  --checkpoint-policy \
  --continuous-mouse \
  --context-frames 129 \
  --batch-size 1 \
  --grad-accum 1 \
  --workers 1 \
  --steps "${STEPS:-18800}" \
  --lr 5e-6 \
  --action-head-lr 1e-5 \
  --lr-schedule constant \
  --mouse-loss-weight .1 \
  --tail-horizon-weight 1 \
  --sampling index \
  --precision bf16 \
  --save-every 100 \
  --snapshot-every 500 \
  --validate-every 100 \
  --val-batches 64 \
  --seed 20261124

