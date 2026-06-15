#!/bin/bash
#SBATCH --job-name=sigreg_deter
#SBATCH --comment="SIGReg-deter Phase2: SIGReg on continuous deter+logits WITH decoder"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/sigreg_deter_p2_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/sigreg_deter_p2_%j.err

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
echo "SIGReg-deter: continuous deter+logits WITH decoder"
echo "Start: $(date)"
echo "=========================================="

python -u main.py \
    --phase phase2 \
    --resume latest \
    --reg sigreg \
    --sigreg-target deter+logits \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/sigreg_deter_p2 \
    --bev-size 300 \
    --batch-size 32 \
    --total-steps 500000 \
    --sigreg-lambda 0.1 \
    --wandb-run-name sigreg_deter_continuous_p2_s42 \
    --preload \
    --seed 42

echo "Done: $(date)"
