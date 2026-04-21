#!/bin/bash

set -euo pipefail

DATA_DIR=${NAVSIM_DATA_DIR:-/share/home/u23516/code/navsim_mini/mini_navsim_logs/mini}
PATTERN=${NAVSIM_PATTERN:-*.pkl}
OUTDIR=${OUTDIR:-/share/home/u23516/code/meta_dreamer-sub/dreamer/logs_navsim_dreamer_posterior_cfm}
RUN_NAME=${RUN_NAME:-navsim_mini_dreamer_posterior_cfm}
MAX_FILES=${MAX_FILES:-0}
MAX_SEQUENCES=${MAX_SEQUENCES:-0}
CONTEXT_LEN=${CONTEXT_LEN:-8}
HORIZON=${HORIZON:-8}
SAMPLE_EVERY=${SAMPLE_EVERY:-2}
DETER_DIM=${DETER_DIM:-512}
STOCH_DIM=${STOCH_DIM:-64}
HIDDEN=${HIDDEN:-512}
WORLD_EPOCHS=${WORLD_EPOCHS:-300}
CFM_EPOCHS=${CFM_EPOCHS:-300}
BATCH_SIZE=${BATCH_SIZE:-128}
LR=${LR:-3e-4}
KL_SCALE=${KL_SCALE:-0.05}
FREE_NATS=${FREE_NATS:-0.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-6}
VAL_RATIO=${VAL_RATIO:-0.1}
SEED=${SEED:-0}
DEVICE=${DEVICE:-auto}
SAVE_EVERY=${SAVE_EVERY:-10}
GRAD_CLIP=${GRAD_CLIP:-10.0}

cd /share/home/u23516/code/meta_dreamer-sub

python3 -u dreamer/tools/navsim_dreamer_posterior_cfm_pretrain.py \
  --data_dir "$DATA_DIR" \
  --pattern "$PATTERN" \
  --outdir "$OUTDIR" \
  --run_name "$RUN_NAME" \
  --max_files "$MAX_FILES" \
  --max_sequences "$MAX_SEQUENCES" \
  --context_len "$CONTEXT_LEN" \
  --horizon "$HORIZON" \
  --sample_every "$SAMPLE_EVERY" \
  --deter_dim "$DETER_DIM" \
  --stoch_dim "$STOCH_DIM" \
  --hidden "$HIDDEN" \
  --world_epochs "$WORLD_EPOCHS" \
  --cfm_epochs "$CFM_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --kl_scale "$KL_SCALE" \
  --free_nats "$FREE_NATS" \
  --weight_decay "$WEIGHT_DECAY" \
  --val_ratio "$VAL_RATIO" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --grad_clip "$GRAD_CLIP" \
  --save_every "$SAVE_EVERY"
