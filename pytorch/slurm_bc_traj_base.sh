#!/bin/bash
#SBATCH --job-name=bc_traj_base
#SBATCH --comment="BC 8000: Phase-as-Input on Baseline WM (step8000)."
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_traj_base_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_traj_base_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== BC 8000: Baseline WM step8000 (phase-as-input) ==="
echo "Start: $(date)"

python -u main.py \
    --phase bc \
    --resume /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_traj_base/checkpoint_step8000.pt \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_traj_base \
    --bev-size 300 \
    --bev-downsample cnn \
    --cnn-factor 2 \
    --batch-size 16 \
    --total-steps 8000 \
    --seed 42 \
    --wandb-run-name bc_traj_base_phase_input_s42

echo "Done: $(date)"
