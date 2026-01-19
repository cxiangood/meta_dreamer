#!/bin/bash
#SBATCH --job-name=dreamer_metadrive
#SBATCH --comment="DreamerV3 / ActorCritic MetaDrive"
#SBATCH --partition=A800
#SBATCH --time=0-2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64g
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/logs/%x_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1504047409@qq.com

set -euo pipefail

# Ensure log directory exists for Slurm output
mkdir -p /share/home/u23516/code/meta_dreamer-main/logs

# -------------------------
# Edit these to match your environment
# -------------------------
# load modules or conda environment
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate py311

# Project root (adjust if needed)
PROJECT_ROOT="/share/home/u23516/code/meta_dreamer-main"
cd "$PROJECT_ROOT" || exit 1

# JAX / GPU tuning (adjust mem fraction as needed)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_CPP_MIN_LOG_LEVEL=1

# Optional: force CPU by uncommenting
# export JAX_PLATFORM_NAME=cpu

# Logdir
LOGDIR_BASE="$PROJECT_ROOT/dreamer/logs_metadrive"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="$LOGDIR_BASE/$TIMESTAMP"
mkdir -p "$LOGDIR"

# IMAGE size env (default 64x64)
if [ -z "${METADRIVE_IMAGE_SIZE+x}" ]; then
  METADRIVE_IMAGE_SIZE="64,64"
fi
IMAGE_SIZE=$(echo "$METADRIVE_IMAGE_SIZE" | tr 'xX ' ',' | sed 's/,,/,/g')
W=$(echo "$IMAGE_SIZE" | cut -d, -f1)
H=$(echo "$IMAGE_SIZE" | cut -d, -f2)
if [ -z "$H" ]; then H="$W"; fi
ENV_METADRIVE_SIZE_ARG="--env.metadrive.size=${W},${H}"

# Which runner: "dreamer" or "actor_critic"
RUN_MODE="${RUN_MODE:-dreamer}"

# Optional tuning for waypoint reward / throttle bias
# export METADRIVE_WAYPOINT_REWARD='10.0'
# export METADRIVE_WAYPOINT_REACH_THRESH='2.0'
# export METADRIVE_THROTTLE_BIAS='0.0'
# export METADRIVE_RENDER='0'   # set to '1' to enable render

echo "Running mode: 'actor_critic'"
echo "Logdir: $LOGDIR"

if [ "$RUN_MODE" = "dreamer" ]; then
  python3 dreamer/dreamerv3/main.py \
    --configs metadrive_lane_keeping \
    --logdir "$LOGDIR" \
    --run.log_every 120 \
    --run.report_every 300 \
    --run.save_every 900 \
    --run.steps 1e5 \
    --run.envs 1 \
    --run.eval_envs 1 \
    --batch_size 16 \
    $ENV_METADRIVE_SIZE_ARG \
    --jax.platform cuda \
    --jax.train_devices 0 \
    --jax.policy_devices 0

elif [ "$RUN_MODE" = "actor_critic" ]; then
  # run the simple JAX actor-critic trainer added earlier
  python3 train_actor_critic.py \
    --logdir "$LOGDIR" \
    --steps 100000 \
    --rollout_steps 128 \
    $ENV_METADRIVE_SIZE_ARG
else
  echo "Unknown RUN_MODE: $RUN_MODE" >&2
  exit 2
fi

# Optional: print hint for tensorboard
echo "Run tensorboard: tensorboard --logdir $LOGDIR_BASE"