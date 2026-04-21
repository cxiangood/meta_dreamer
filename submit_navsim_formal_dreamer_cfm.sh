#!/usr/bin/env bash
#SBATCH --job-name=navsim_formal_cfm
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/share/home/u23516/code/meta_dreamer-sub/logs/%x_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-sub/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1504047409@qq.com

set -euo pipefail

ROOT_DIR="/share/home/u23516/code/meta_dreamer-sub"
mkdir -p "${ROOT_DIR}/logs"

export ROOT_DIR
export RUN_NAME="${RUN_NAME:-navsim_mini_cached_feature_dreamer_cfm_$(date +%Y%m%d_%H%M%S)}"
export OUTDIR="${OUTDIR:-${ROOT_DIR}/dreamer/logs_navsim_cached_feature_dreamer_cfm}"

MAIN_LOG="${ROOT_DIR}/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
MONITOR_LOG="${ROOT_DIR}/logs/monitor_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
RUN_DIR="${OUTDIR}/${RUN_NAME}"

monitor_loop() {
  while true; do
    sleep 300
    {
      echo "===== $(date '+%Y-%m-%d %H:%M:%S') job=${SLURM_JOB_ID} run=${RUN_NAME} ====="
      squeue -j "${SLURM_JOB_ID}" || true
      echo "--- stdout tail ---"
      tail -n 80 "${MAIN_LOG}" 2>/dev/null || true
      echo "--- checkpoints ---"
      find "${RUN_DIR}" -maxdepth 1 -type f \( -name '*.pt' -o -name '*.json' \) -printf '%TY-%Tm-%Td %TH:%TM %p\n' 2>/dev/null | sort || true
      echo
    } >> "${MONITOR_LOG}"
    echo "[Monitor] $(date '+%Y-%m-%d %H:%M:%S') wrote ${MONITOR_LOG}"
  done
}

monitor_loop &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

echo "[Submit] job=${SLURM_JOB_ID} run=${RUN_NAME}"
echo "[Submit] outdir=${RUN_DIR}"
echo "[Submit] monitor=${MONITOR_LOG}"

bash "${ROOT_DIR}/run_navsim_formal_dreamer_cfm.sh"
