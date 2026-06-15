#!/bin/bash
#SBATCH --job-name=df_phase_cond
#SBATCH --comment="Plan A: Phase-Conditional RSSM + JEPA(k=125) + TrajHead + Speed"
#SBATCH --partition=A800
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_phase_cond_p2_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_phase_cond_p2_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs
mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_phase_cond

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== Plan A: Phase-Conditional RSSM + JEPA(k=125=5s) + TrajHead + Speed ==="
echo "Phase-Conditional: 3 independent dynamics heads gated by GT phase"
echo "JEPA k=125: predict feature 5 seconds ahead"
echo "Start: $(date)"

python -u main.py \
    --phase phase2 \
    --reg sigreg \
    --sigreg-target deter+logits \
    --use-decoder False \
    --barlow-lambda 0.005 \
    --barlow-k 1 \
    --bev-downsample cnn \
    --cnn-factor 2 \
    --batch-size 4 \
    --batch-length 130 \
    --use-speed-head \
    --speed-head-weight 1.0 \
    --use-jepa \
    --jepa-weight 0.1 \
    --jepa-k 125 \
    --use-traj-head \
    --traj-head-weight 0.1 \
    --traj-horizon 10 \
    --rssm-phase-conditional \
    --use-phase-head \
    --phase-head-weight 1.0 \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_phase_cond \
    --log-every 50 \
    --wandb-run-name "df_phase_cond_p2"

echo "Done: $(date)"
