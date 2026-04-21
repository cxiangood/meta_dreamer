#!/bin/bash

set -euo pipefail

DATA_DIR=${NAVSIM_DATA_DIR:-/share/home/u23516/code/navsim_mini/mini_navsim_logs/mini}
PATTERN=${NAVSIM_PATTERN:-*.pkl}
OUTDIR=${OUTDIR:-/share/home/u23516/code/meta_dreamer-sub/dreamer/logs_navsim_cfm}
RUN_NAME=${RUN_NAME:-navsim_mini_cfm_latent}
MAX_FILES=${MAX_FILES:-0}
MAX_SAMPLES=${MAX_SAMPLES:-0}
HORIZON=${HORIZON:-8}
STRIDE=${STRIDE:-1}
SAMPLE_EVERY=${SAMPLE_EVERY:-2}
LATENT_DIM=${LATENT_DIM:-12}
HIDDEN=${HIDDEN:-256}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-512}
LR=${LR:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}
VAL_RATIO=${VAL_RATIO:-0.1}
SEED=${SEED:-0}
SAVE_EVERY=${SAVE_EVERY:-10}
BACKEND=${BACKEND:-auto}
DEVICE=${DEVICE:-auto}

cd /share/home/u23516/code/meta_dreamer-sub

python3 -u dreamer/tools/navsim_cfm_latent_pretrain.py \
  --data_dir "$DATA_DIR" \
  --pattern "$PATTERN" \
  --outdir "$OUTDIR" \
  --run_name "$RUN_NAME" \
  --max_files "$MAX_FILES" \
  --max_samples "$MAX_SAMPLES" \
  --horizon "$HORIZON" \
  --stride "$STRIDE" \
  --sample_every "$SAMPLE_EVERY" \
  --latent_dim "$LATENT_DIM" \
  --hidden "$HIDDEN" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --val_ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --save_every "$SAVE_EVERY" \
  --backend "$BACKEND" \
  --device "$DEVICE"
