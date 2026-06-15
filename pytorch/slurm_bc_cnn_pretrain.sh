#!/bin/bash
#SBATCH --job-name=bc_cnn
#SBATCH --comment="BC pretrain: clone GT actions on CNN P2 step4000 WM (300x300 → CNN → 150x150)"
#SBATCH --partition=A800
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=48G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_cnn_pretrain_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_cnn_pretrain_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== BC Pretrain: CNN WM (300×300→CNN→150×150) + Sequence BC ==="
echo "WM ckpt: df_sig_cnn_p2/checkpoint_step4000.pt"
echo "Seq BC: 50 frames/step, RSSM observes REAL actions (not dummy)"
echo "2000 steps × 800 frames/step = 1.6M frames total"
echo "Start: $(date)"

python -u main.py \
    --phase bc \
    --reg sigreg \
    --sigreg-target deter+logits \
    --use-decoder False \
    --barlow-lambda 0.005 \
    --barlow-k 1 \
    --bev-downsample cnn \
    --cnn-factor 2 \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_cnn \
    --bev-size 300 \
    --batch-size 16 \
    --total-steps 2000 \
    --sigreg-lambda 0.1 \
    --wandb-run-name bc_cnn_seq_s42 \
    --resume /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_cnn_p2/checkpoint_step4000.pt \
    --seed 42

echo "Done: $(date)"
