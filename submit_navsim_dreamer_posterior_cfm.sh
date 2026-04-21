#!/bin/bash
#SBATCH --job-name=navsim_dreamer_cfm
#SBATCH --comment="NAVSIM Dreamer posterior latent CFM pretraining"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64g
#SBATCH --output=/share/home/u23516/code/meta_dreamer-sub/logs/%x_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-sub/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1504047409@qq.com

set -euo pipefail

mkdir -p /share/home/u23516/code/meta_dreamer-sub/logs

if [ -f /share/home/u23516/miniforge3/etc/profile.d/conda.sh ]; then
  source /share/home/u23516/miniforge3/etc/profile.d/conda.sh
  conda activate metadrive || true
fi

export PYTHONPATH="/share/home/u23516/code/meta_dreamer-sub/dreamer:${PYTHONPATH:-}"
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

cd /share/home/u23516/code/meta_dreamer-sub

RUN_NAME=${RUN_NAME:-navsim_mini_dreamer_posterior_cfm_$(date +"%Y%m%d_%H%M%S")}
export RUN_NAME

MONITOR_INTERVAL=${MONITOR_INTERVAL:-300}
SLURM_OUT="/share/home/u23516/code/meta_dreamer-sub/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
SLURM_ERR="/share/home/u23516/code/meta_dreamer-sub/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
MONITOR_LOG="/share/home/u23516/code/meta_dreamer-sub/logs/monitor_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
RUN_OUTDIR="${OUTDIR:-/share/home/u23516/code/meta_dreamer-sub/dreamer/logs_navsim_dreamer_posterior_cfm}/${RUN_NAME}"

monitor_logs() {
  while kill -0 "$1" 2>/dev/null; do
    sleep "$MONITOR_INTERVAL"
    echo "[Monitor $(date '+%F %T')] job=${SLURM_JOB_ID} run=${RUN_NAME} details=${MONITOR_LOG}"
    {
      echo "===== Monitor $(date '+%F %T') job=${SLURM_JOB_ID} run=${RUN_NAME} ====="
      echo "run_outdir=${RUN_OUTDIR}"
      if [ -d "$RUN_OUTDIR" ]; then
        find "$RUN_OUTDIR" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort | tail -8
      fi
      if [ -f "$SLURM_OUT" ]; then
        echo "----- stdout tail -----"
        tail -30 "$SLURM_OUT"
      fi
      if [ -s "$SLURM_ERR" ]; then
        echo "----- stderr tail -----"
        tail -30 "$SLURM_ERR"
      fi
    } >> "$MONITOR_LOG"
  done
}

bash /share/home/u23516/code/meta_dreamer-sub/run_navsim_dreamer_posterior_cfm.sh &
TRAIN_PID=$!
monitor_logs "$TRAIN_PID" &
MONITOR_PID=$!
wait "$TRAIN_PID"
STATUS=$?
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
exit "$STATUS"
