#!/bin/bash
#SBATCH --job-name=bc_pretrain
#SBATCH --comment="BC pretraining: clone GT actions in DF+SIG WM feature space (from best P2 ckpt)"
#SBATCH --partition=A800
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_pretrain_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_pretrain_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== BC Pretraining: DF+SIG + GT action cloning ==="
echo "WM ckpt: df_sigreg_p2/checkpoint_best.pt"
echo "Start: $(date)"

python -u main.py \
    --phase bc \
    --reg sigreg \
    --sigreg-target deter+logits \
    --use-decoder False \
    --barlow-lambda 0.005 \
    --barlow-k 1 \
    --bev-downsample bilinear \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_df_sigreg \
    --bev-size 64 \
    --batch-size 64 \
    --total-steps 50000 \
    --sigreg-lambda 0.1 \
    --wandb-run-name bc_df_sigreg_s42 \
    --resume /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2/checkpoint_best.pt \
    --seed 42

echo "Done: $(date)"
