#!/bin/bash

set -euo pipefail

DATA_DIR=${NAVSIM_DATA_DIR:-/data/navsim}
PATTERN=${NAVSIM_PATTERN:-*.npz}
MAX_EPISODES=${MAX_EPISODES:-0}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-3e-4}
VAL_RATIO=${VAL_RATIO:-0.1}
SEED=${SEED:-0}
SAVE_PATH=${SAVE_PATH:-dreamer/logs_navsim/navsim_instruction_v1.params}

python3 dreamer/tools/navsim_instruction_v1.py \
  --data_dir "$DATA_DIR" \
  --pattern "$PATTERN" \
  --max_episodes "$MAX_EPISODES" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --val_ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --save_path "$SAVE_PATH"

