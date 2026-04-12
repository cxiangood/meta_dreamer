#!/bin/bash

#SBATCH --job-name=il_rl
#SBATCH --comment="DreamerV3 MetaDrive IL+RL"
#SBATCH --partition=A800
#SBATCH --time=0-5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64g
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/logs/%x_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1504047409@qq.com

# Ensure log directory exists for Slurm output
mkdir -p logs

# Load your environment here (module or conda); placeholder below
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate python311
# Load runtime environment.
source /share/home/u23516/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
export PYTHONPATH="/share/home/u23516/code/meta_dreamer-main/dreamer:${PYTHONPATH}"

# Move to project root
cd /share/home/u23516/code/meta_dreamer-main || exit 1

# Reproducibility settings
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Recommended: set JAX to single A800 and avoid prealloc issues
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_CPP_MIN_LOG_LEVEL=1

# Logdir base
LOGDIR_BASE="/share/home/u23516/code/meta_dreamer-main/dreamer/logs_metadrive"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_TAG=${EXPERIMENT_TAG:-topconf}
SEED=${SEED:-0}
LOGDIR="$LOGDIR_BASE/${TIMESTAMP}_${EXPERIMENT_TAG}_s${SEED}"
mkdir -p "$LOGDIR"

# Image size override (format: WxH, W,H, or W x H). Default matches the config [64,64].
if [ -z "${METADRIVE_IMAGE_SIZE+x}" ]; then
  METADRIVE_IMAGE_SIZE="64,64"
fi
IMAGE_SIZE=$(echo "$METADRIVE_IMAGE_SIZE" | tr 'xX ' ',' | sed 's/,,/,/g')
W=$(echo "$IMAGE_SIZE" | cut -d, -f1)
H=$(echo "$IMAGE_SIZE" | cut -d, -f2)
if [ -z "$H" ]; then H="$W"; fi
ENV_METADRIVE_SIZE_ARG="--env.metadrive.size=${W},${H}"

# Runtime overrides for top-level ablations
RUN_STEPS=${RUN_STEPS:-2e6}
RUN_ENVS=${RUN_ENVS:-1}
RUN_EVAL_ENVS=${RUN_EVAL_ENVS:-1}
BATCH_SIZE=${BATCH_SIZE:-16}
TRAIN_RATIO=${TRAIN_RATIO:-48}

EXPERT_PROB_INIT=${EXPERT_PROB_INIT:-0.95}
EXPERT_PROB_FINAL=${EXPERT_PROB_FINAL:-0.30}
EXPERT_DECAY_STEPS=${EXPERT_DECAY_STEPS:-800000}
EXPERT_DECAY_TYPE=${EXPERT_DECAY_TYPE:-cosine}
ACTION_THRESHOLD=${ACTION_THRESHOLD:-0.15}

EXPERT_HEADS=${EXPERT_HEADS:-1}
EXPERT_MODES=${EXPERT_MODES:-1}
EXPERT_TRAJ_HORIZON=${EXPERT_TRAJ_HORIZON:-8}
EXPERT_TRAJ_DT=${EXPERT_TRAJ_DT:-0.2}

RISK_THRESHOLD=${RISK_THRESHOLD:-0.40}
RISK_SPEED_WEIGHT=${RISK_SPEED_WEIGHT:-0.20}
RISK_LATERAL_WEIGHT=${RISK_LATERAL_WEIGHT:-0.45}
RISK_HEADING_WEIGHT=${RISK_HEADING_WEIGHT:-0.25}
RISK_DISAGREEMENT_WEIGHT=${RISK_DISAGREEMENT_WEIGHT:-0.10}
RISK_MAX_SPEED=${RISK_MAX_SPEED:-16.0}
RISK_MAX_LATERAL=${RISK_MAX_LATERAL:-2.0}

python3 dreamer/dreamerv3/main.py \
  --configs metadrive_lane_keeping_dagger \
  --logdir "$LOGDIR" \
  --seed "$SEED" \
  --run.log_every 120 \
  --run.report_every 300 \
  --run.save_every 900 \
  --run.steps "$RUN_STEPS" \
  --run.envs "$RUN_ENVS" \
  --run.eval_envs "$RUN_EVAL_ENVS" \
  --run.train_ratio "$TRAIN_RATIO" \
  --batch_size "$BATCH_SIZE" \
  --env.metadrive.expert_prob_init "$EXPERT_PROB_INIT" \
  --env.metadrive.expert_prob_final "$EXPERT_PROB_FINAL" \
  --env.metadrive.expert_decay_steps "$EXPERT_DECAY_STEPS" \
  --env.metadrive.expert_decay_type "$EXPERT_DECAY_TYPE" \
  --env.metadrive.action_threshold "$ACTION_THRESHOLD" \
  --env.metadrive.expert_heads "$EXPERT_HEADS" \
  --env.metadrive.expert_modes "$EXPERT_MODES" \
  --env.metadrive.expert_traj_horizon "$EXPERT_TRAJ_HORIZON" \
  --env.metadrive.expert_traj_dt "$EXPERT_TRAJ_DT" \
  --env.metadrive.risk_threshold "$RISK_THRESHOLD" \
  --env.metadrive.risk_speed_weight "$RISK_SPEED_WEIGHT" \
  --env.metadrive.risk_lateral_weight "$RISK_LATERAL_WEIGHT" \
  --env.metadrive.risk_heading_weight "$RISK_HEADING_WEIGHT" \
  --env.metadrive.risk_disagreement_weight "$RISK_DISAGREEMENT_WEIGHT" \
  --env.metadrive.risk_max_speed "$RISK_MAX_SPEED" \
  --env.metadrive.risk_max_lateral "$RISK_MAX_LATERAL" \
  $ENV_METADRIVE_SIZE_ARG \
  --jax.prealloc False \
  --jax.debug False
TRAIN_EXIT_CODE=$?

# Auto analysis report (CSV/JSON/MD/HTML), best-effort.
ANALYSIS_OUT="${LOGDIR}/analysis"
python3 dreamer/tools/analyze_experiments.py \
  --roots "$LOGDIR_BASE" "$LOGDIR" \
  --outdir "$ANALYSIS_OUT" \
  --topk 20 || true
echo "[Analysis] Report generated under: ${ANALYSIS_OUT}"
echo "[Analysis] Open: ${ANALYSIS_OUT}/summary.html"

exit $TRAIN_EXIT_CODE

# TensorBoard hint (optional):
# srun --ntasks=1 --cpus-per-task=2 --gres=gpu:0 --time=0-5:00:00 \
#   tensorboard --logdir "$LOGDIR_BASE" --port 6007 --host 0.0.0.0
