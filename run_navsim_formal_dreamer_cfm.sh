#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/share/home/u23516/code/meta_dreamer-sub}"
DATA_DIR="${DATA_DIR:-/share/home/u23516/code/navsim_mini/mini_navsim_logs/mini}"
IMAGE_ROOT="${IMAGE_ROOT:-/share/home/u23516/code/navsim_mini/mini_sensor_blobs/mini}"
OUTDIR="${OUTDIR:-${ROOT_DIR}/dreamer/logs_navsim_cached_feature_dreamer_cfm}"
RUN_NAME="${RUN_NAME:-navsim_mini_cached_feature_dreamer_cfm_$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR="${CACHE_DIR:-${ROOT_DIR}/dreamer/navsim_feature_cache/formal_cam_f0_s96_g12}"

CAMERA="${CAMERA:-CAM_F0}"
IMAGE_SIZE="${IMAGE_SIZE:-96}"
GRID_SIZE="${GRID_SIZE:-12}"
CONTEXT_LEN="${CONTEXT_LEN:-8}"
HORIZON="${HORIZON:-8}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2}"
MAX_FILES="${MAX_FILES:-0}"
MAX_SEQUENCES="${MAX_SEQUENCES:-0}"

BATCH_SIZE="${BATCH_SIZE:-512}"
WORLD_EPOCHS="${WORLD_EPOCHS:-120}"
CFM_EPOCHS="${CFM_EPOCHS:-120}"
DETER_DIM="${DETER_DIM:-512}"
STOCH_DIM="${STOCH_DIM:-64}"
HIDDEN="${HIDDEN:-512}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LR="${LR:-3e-4}"
KL_SCALE="${KL_SCALE:-0.05}"
VISUAL_RECON_SCALE="${VISUAL_RECON_SCALE:-0.5}"
SAVE_EVERY="${SAVE_EVERY:-10}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-10}"
REBUILD_CACHE="${REBUILD_CACHE:-0}"

cd "${ROOT_DIR}"

if [ -f /share/home/u23516/miniforge3/etc/profile.d/conda.sh ]; then
  source /share/home/u23516/miniforge3/etc/profile.d/conda.sh
  conda activate metadrive
fi

EXTRA_ARGS=()
if [ "${REBUILD_CACHE}" = "1" ]; then
  EXTRA_ARGS+=(--rebuild_cache)
fi

python3 -u dreamer/tools/navsim_cached_feature_dreamer_cfm_pretrain.py \
  --data_dir "${DATA_DIR}" \
  --image_root "${IMAGE_ROOT}" \
  --camera "${CAMERA}" \
  --image_size "${IMAGE_SIZE}" \
  --grid_size "${GRID_SIZE}" \
  --cache_dir "${CACHE_DIR}" \
  --outdir "${OUTDIR}" \
  --run_name "${RUN_NAME}" \
  --max_files "${MAX_FILES}" \
  --max_sequences "${MAX_SEQUENCES}" \
  --context_len "${CONTEXT_LEN}" \
  --horizon "${HORIZON}" \
  --sample_every "${SAMPLE_EVERY}" \
  --deter_dim "${DETER_DIM}" \
  --stoch_dim "${STOCH_DIM}" \
  --hidden "${HIDDEN}" \
  --world_epochs "${WORLD_EPOCHS}" \
  --cfm_epochs "${CFM_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --kl_scale "${KL_SCALE}" \
  --visual_recon_scale "${VISUAL_RECON_SCALE}" \
  --save_every "${SAVE_EVERY}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --log_every_batches "${LOG_EVERY_BATCHES}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
