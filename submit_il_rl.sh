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

# Move to project root
cd /share/home/u23516/code/meta_dreamer-main || exit 1

# Recommended: set JAX to single A800 and avoid prealloc issues
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_CPP_MIN_LOG_LEVEL=1

# Logdir base
LOGDIR_BASE="/share/home/u23516/code/meta_dreamer-main/dreamer/logs_metadrive"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="$LOGDIR_BASE/$TIMESTAMP"
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

python3 dreamer/dreamerv3/main.py \
  --configs metadrive_lane_keeping_dagger \
  --logdir "$LOGDIR" \
  --run.log_every 120 \
  --run.report_every 300 \
  --run.save_every 900 \
  --run.steps 2e6 \
  --run.envs 1 \
  --run.eval_envs 1 \
  --batch_size 16 \
  $ENV_METADRIVE_SIZE_ARG \
  --jax.prealloc false \
  --jax.debug false

# TensorBoard hint (optional):
# srun --ntasks=1 --cpus-per-task=2 --gres=gpu:0 --time=0-5:00:00 \
#   tensorboard --logdir "$LOGDIR_BASE" --port 6007 --host 0.0.0.0
