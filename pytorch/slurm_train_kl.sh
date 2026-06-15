#!/bin/bash
#SBATCH --job-name=kl_p2
#SBATCH --comment="DreamerV3(KL) Phase2: Offline WM pretraining"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/kl_p2_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/kl_p2_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive

cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "=========================================="

python -u main.py \
    --phase phase2 \
    --resume latest \
    --reg kl \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/kl_p2 \
    --bev-size 300 \
    --batch-size 32 \
    --total-steps 500000 \
    --kl-beta 1.0 \
    --wandb-run-name kl_p2_s42 \
    --preload \
    --seed 42

echo "Done: $(date)"
