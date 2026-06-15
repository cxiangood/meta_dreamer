#!/bin/bash
#SBATCH --job-name=bc_topo
#SBATCH --comment="BC+Topo: clone GT actions + predict merge phase (3-class) in actor feature space"
#SBATCH --partition=A800
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=32G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_topo_pretrain_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_topo_pretrain_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== BC+Topo: Behavior Cloning + Topology-Guided Phase Head ==="
echo "WM ckpt: df_sigreg_p2/checkpoint_best.pt"
echo "Phase head: 3-class in actor (ramp/merge/main)"
echo "Start: $(date)"

python -u main.py \
    --phase bc \
    --reg sigreg \
    --sigreg-target deter+logits \
    --use-decoder False \
    --barlow-lambda 0.005 \
    --barlow-k 1 \
    --bev-downsample bilinear \
    --use-phase-head \
    --phase-head-weight 1.0 \
    --merge-zone-frames 20 \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_df_sigreg_topo \
    --bev-size 64 \
    --batch-size 64 \
    --total-steps 50000 \
    --sigreg-lambda 0.1 \
    --wandb-run-name bc_topo_s42 \
    --resume /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2/checkpoint_best.pt \
    --seed 42

echo "Done: $(date)"
