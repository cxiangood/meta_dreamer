#!/bin/bash
#SBATCH --job-name=df_sig_p3
#SBATCH --comment="Phase3 imagination training on DF+SIG checkpoint (WM frozen, AC only)"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_p3_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_p3_%j.err

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
echo "Phase 3: Imagination AC training (DF+SIG WM frozen)"
echo "Start: $(date)"
echo "=========================================="

python -u main.py \
    --phase phase3 \
    --reg sigreg \
    --sigreg-target deter+logits \
    --sigreg-lambda 0.1 \
    --use-decoder False \
    --barlow-lambda 0.005 \
    --barlow-k 1 \
    --resume /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2/checkpoint_latest.pt \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_p3 \
    --bev-size 300 \
    --batch-size 16 \
    --total-steps 200000 \
    --wandb-run-name df_sig_p3_s42 \
    --seed 42

echo "Done: $(date)"
