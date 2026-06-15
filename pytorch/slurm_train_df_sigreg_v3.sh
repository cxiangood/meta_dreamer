#!/bin/bash
#SBATCH --job-name=df_sigreg
#SBATCH --comment="DF+SIGReg Phase2 resume from step2000 (5sxzujvz config)"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2_%j.err

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
echo "DF+SIGReg Phase2: resume from step2000 (barlow=0.005, bs=32)"
echo "Start: $(date)"
echo "=========================================="

python -u main.py     --phase phase2     --resume latest     --reg sigreg     --sigreg-target deter+logits     --use-decoder False     --barlow-lambda 0.005     --barlow-k 1     --data-dir /share/home/u23516/data/exid_dreamer_data     --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2     --bev-size 64     --batch-size 32     --total-steps 500000     --sigreg-lambda 0.1     --seed 42

echo "Done: $(date)"
