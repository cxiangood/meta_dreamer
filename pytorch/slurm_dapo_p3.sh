#!/bin/bash
#SBATCH --job-name=dapo_p3
#SBATCH --comment="DAPO P3 v1: group-based PG, no critic, K=8"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/dapo_p3_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/dapo_p3_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== DAPO P3 v1 (group-based PG, K=8, no critic) ==="
echo "Start: $(date)"

python -u main.py \
    --phase phase3 \
    --dapo --dapo-k 8 \
    --reg sigreg \
    --sigreg-target deter+logits --sigreg-lambda 0.1 \
    --use-decoder False --barlow-lambda 0.005 --barlow-k 1 \
    --resume /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2/checkpoint_latest.pt \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/dapo_p3 \
    --bev-size 300 --batch-size 16 --total-steps 200000 \
    --wandb-run-name df_sig_dapo_p3_s42_v1 --seed 42

echo "Done: $(date)"
