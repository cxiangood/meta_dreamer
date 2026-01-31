#!/bin/bash

#SBATCH --job-name=cfm_lanekeep
#SBATCH --comment="Flow Matching + DreamerV3 Lane Keeping"
#SBATCH --partition=A800
#SBATCH --time=0-8:00:00
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
# Conditional Flow Matching (CFM) + World Model Configuration
# ============================================================================
# This script trains a Flow Matching policy head with DreamerV3 world model
# for the lane keeping task in MetaDrive.
#
# Flow Matching advantages:
#   - Supports multi-modal action distributions
#   - Simpler training than diffusion models
#   - Faster inference with fewer ODE steps
#   - Better for complex driving scenarios
#
# Options:
#   USE_SIGLIP=true   - Use SIGLIP 2 vision encoder
#   USE_SIGLIP=false  - Use simple CNN encoder
#   USE_ACTION_CHUNK  - Enable action chunking for temporal consistency
# ============================================================================

USE_SIGLIP=false
USE_ACTION_CHUNK=false

# JAX/GPU settings
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_CPP_MIN_LOG_LEVEL=1
export JAX_PLATFORMS=cuda,cpu

# Logdir setup
LOGDIR_BASE="/share/home/u23516/code/meta_dreamer-main/dreamer/logs_cfm"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="$LOGDIR_BASE/$TIMESTAMP"
mkdir -p "$LOGDIR"

# Git commit for tracking
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
export GIT_COMMIT

# ============================================================================
# Flow Matching Configuration
# ============================================================================
FLOW_HIDDEN=512        # Hidden dimension of velocity network
FLOW_LAYERS=4          # Number of MLP/Transformer layers
FLOW_STEPS=10          # ODE integration steps at inference
CHUNK_SIZE=1           # Action chunk size (1 = no chunking)

if [ "$USE_ACTION_CHUNK" = "true" ]; then
    CHUNK_SIZE=4
    echo "Action chunking enabled: chunk_size=$CHUNK_SIZE"
fi

# Select encoder configuration
if [ "$USE_SIGLIP" = "true" ]; then
    echo "=========================================="
    echo "Running CFM with SIGLIP 2 + CNN fusion"
    echo "Image size: 256x256"
    echo "=========================================="
    CONFIG="vla_ilrl_fusion"
    IMAGE_SIZE="256,256"
    SIGLIP_PATH="../siglip2-so400m-patch16-256"
    ENCODER_ARGS="--agent.enc.siglip_cnn.siglip_path=$SIGLIP_PATH"
else
    echo "=========================================="
    echo "Running CFM with simple CNN encoder"
    echo "Image size: 64x64"
    echo "=========================================="
    CONFIG="metadrive_lane_keeping"
    IMAGE_SIZE="64,64"
    ENCODER_ARGS=""
fi

# Parse image size
W=$(echo "$IMAGE_SIZE" | cut -d, -f1)
H=$(echo "$IMAGE_SIZE" | cut -d, -f2)
ENV_SIZE_ARG="--env.metadrive.size=${W},${H}"

# Flow Matching arguments
FLOW_ARGS="--agent.policy_head.typ=flow"
FLOW_ARGS="$FLOW_ARGS --agent.policy_head.flow.hidden=$FLOW_HIDDEN"
FLOW_ARGS="$FLOW_ARGS --agent.policy_head.flow.layers=$FLOW_LAYERS"
FLOW_ARGS="$FLOW_ARGS --agent.policy_head.flow.inference_steps=$FLOW_STEPS"
FLOW_ARGS="$FLOW_ARGS --agent.policy_head.flow.chunk_size=$CHUNK_SIZE"
FLOW_ARGS="$FLOW_ARGS --agent.policy_head.flow.use_visual_residual=True"
FLOW_ARGS="$FLOW_ARGS --agent.policy_head.flow.use_transformer=False"

echo ""
echo "=========================================="
echo "Conditional Flow Matching Configuration"
echo "=========================================="
echo "Logdir: $LOGDIR"
echo "Config: $CONFIG"
echo "Git: $GIT_COMMIT"
echo ""
echo "Flow Matching Settings:"
echo "  - Hidden dim: $FLOW_HIDDEN"
echo "  - Layers: $FLOW_LAYERS"
echo "  - Inference steps: $FLOW_STEPS"
echo "  - Chunk size: $CHUNK_SIZE"
echo "=========================================="
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
  $ENCODER_ARGS \
  $FLOW_ARGS

echo ""
echo "=========================================="
echo "Training completed!"
echo "Logs saved to: $LOGDIR"
echo "=========================================="

# TensorBoard hint:
# tensorboard --logdir "$LOGDIR_BASE" --port 6007 --host 0.0.0.0
