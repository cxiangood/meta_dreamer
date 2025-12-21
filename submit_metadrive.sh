#!/bin/bash

#SBATCH --job-name=dreamer_metadrive
#SBATCH --comment="DreamerV3 MetaDrive lane keeping"
#SBATCH --partition=A800
#SBATCH --time=0-02:00:00
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

# Run training (2h wall time enforced by Slurm)
python3 dreamer/dreamerv3/main.py \
  --configs metadrive_lane_keeping \
  --logdir "$LOGDIR" \
  --run.log_every 120 \
  --run.report_every 300 \
  --run.save_every 900 \
  --run.steps 1e6 \
  --run.envs 4 \
  --run.eval_envs 4 \
  --batch_size 8 \
  --jax.platform cuda \
  --jax.train_devices 0 \
  --jax.policy_devices 0

# TensorBoard hint (optional):
# srun --ntasks=1 --cpus-per-task=2 --gres=gpu:0 --time=0-02:00:00 \
#   tensorboard --logdir "$LOGDIR_BASE" --port 6006 --host 0.0.0.0
