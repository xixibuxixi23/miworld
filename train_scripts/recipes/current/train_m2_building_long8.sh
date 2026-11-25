#!/usr/bin/env bash
set -euo pipefail

: "${MIWORLD_ROOT:?set MIWORLD_ROOT to this repository}"
: "${PYTHON_BIN:?set PYTHON_BIN to the training environment python}"
: "${M2_SOURCE_CHECKPOINT:?set the immutable engine-prior parent checkpoint}"
: "${M2_RESUME_CHECKPOINT:?set the complete checkpoint to resume}"
: "${M2_INDEX:?set the train/dev episode index}"
: "${M2_CACHE_ROOT:?set the revision-4 cache root}"
: "${M2_CATALOG_PATH:?set the frozen building-overfit catalog}"
: "${OUTPUT_DIR:?set a new output directory}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29739}"

export PYTHONPATH="$MIWORLD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" -m torch.distributed.run \
  --nnodes="$NNODES" \
  --nproc_per_node="$NPROC_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$MIWORLD_ROOT/train_scripts/train_m2_long_rollout.py" \
  --source-checkpoint "$M2_SOURCE_CHECKPOINT" \
  --resume "$M2_RESUME_CHECKPOINT" \
  --resume-world-size-change \
  --resume-batch-size-change \
  --index "$M2_INDEX" \
  --cache-root "$M2_CACHE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --stop-file "$OUTPUT_DIR/STOP" \
  --allow-train-val-overlap \
  --train-scope full \
  --steps "${STEPS:-100000}" \
  --warmup-steps 0 \
  --lr-warmup-steps 50 \
  --rollout-blocks 8 \
  --bptt-blocks 1 \
  --eval-blocks 8 \
  --batch-size "${BATCH_SIZE:-16}" \
  --eval-batches 3 \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --save-every 100 \
  --log-every 10 \
  --backbone-lr 5e-6 \
  --new-lr 1e-4 \
  --velocity-weight 0.5 \
  --endpoint-weight 1 \
  --stationary-weight 0.25 \
  --residual-penalty 0.01 \
  --event-count-positive-weight 4 \
  --edit-oversample 3 \
  --attack-oversample 2 \
  --threads 2

