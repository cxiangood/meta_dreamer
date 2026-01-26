#!/bin/bash

#SBATCH --job-name=vla_il_rl
#SBATCH --comment="VLA + DreamerV3 IL+RL with SIGLIP 2"
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

# Move to project root
cd /share/home/u23516/code/meta_dreamer-main || exit 1

# ============================================================================
# VLA + World Model Configuration
# ============================================================================
# This script trains a VLA (Vision-Language-Action) + World Model architecture
# combining SIGLIP 2 vision encoder with DreamerV3 world model.
#
# Options:
#   USE_SIGLIP=true   - Use pretrained SIGLIP 2 (requires PyTorch + transformers)
#   USE_SIGLIP=false  - Use pure JAX ViT encoder (no PyTorch dependency)
# ============================================================================

USE_SIGLIP=true
# JAX/GPU settings
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_CPP_MIN_LOG_LEVEL=1
export JAX_PLATFORMS=cuda,cpu

# Logdir setup
LOGDIR_BASE="/share/home/u23516/code/meta_dreamer-main/dreamer/logs_vla"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="$LOGDIR_BASE/$TIMESTAMP"
# LOGDIR="/share/home/u23516/code/meta_dreamer-main/dreamer/logs_vla/20260125_215846"
mkdir -p "$LOGDIR"

# Git commit for tracking
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_COMMIT

# Select configuration based on SIGLIP usage
if [ "$USE_SIGLIP" = "true" ]; then
    echo "=========================================="
    echo "Running VLA with pretrained SIGLIP 2"
    echo "Image size: 256x256"
    echo "=========================================="
  CONFIG="vla_metadrive"
    IMAGE_SIZE="256,256"
    # SIGLIP model path (relative to project root)
    SIGLIP_PATH="../siglip2-so400m-patch16-256"
    EXTRA_ARGS="--agent.enc.siglip.siglip_path=$SIGLIP_PATH"
else
    echo "=========================================="
    echo "Running VLA with pure JAX ViT encoder"
    echo "Image size: 64x64"
    echo "=========================================="
  CONFIG="vla_jax_metadrive"
    IMAGE_SIZE="64,64"
    EXTRA_ARGS=""
fi

# Parse image size
W=$(echo "$IMAGE_SIZE" | cut -d, -f1)
H=$(echo "$IMAGE_SIZE" | cut -d, -f2)
ENV_SIZE_ARG="--env.metadrive.size=${W},${H}"

echo "Logdir: $LOGDIR"
echo "Config: $CONFIG"
echo "Git: $GIT_COMMIT"
echo ""

# Run training
python3 dreamer/dreamerv3/main.py \
  --configs $CONFIG \
  --logdir "$LOGDIR" \
  --run.log_every 120 \
  --run.report_every 300 \
  --run.save_every 900 \
  --run.steps 1e6 \
  --run.envs 1 \
  --run.eval_envs 1 \
  --batch_size 8 \
  $ENV_SIZE_ARG \
  --jax.platform "cuda,cpu" \
  --jax.train_devices 0 \
  --jax.policy_devices 0 \
  $EXTRA_ARGS

echo "=========================================="
echo "Training completed!"
echo "Logs saved to: $LOGDIR"
echo "=========================================="

# TensorBoard hint:
# tensorboard --logdir "$LOGDIR_BASE" --port 6007 --host 0.0.0.0
