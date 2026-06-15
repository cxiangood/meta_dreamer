#!/bin/bash
#SBATCH --job-name=online_dreamer
#SBATCH --comment="Online DreamerV3: interact with exiD loc0/2/4/5/6, train WM+AC mixed"
#SBATCH --partition=A800
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_dreamer_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_dreamer_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs
mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_dreamer

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== Online DreamerV3 Training ==="
echo "WM: df_sig_cnn_jepa_traj/checkpoint_step2000.pt"
echo "AC: bc_jepa_traj/checkpoint_bc_best.pt"
echo "Train locs: 0 2 4 5 6 | Eval locs: 1 3"
echo "Start: $(date)"

python -u train_online_dreamer.py \
    --wm-ckpt /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_cnn_jepa_traj/checkpoint_step2000.pt \
    --ac-ckpt /share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_jepa_traj/checkpoint_bc_best.pt \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --npz-dir /share/home/u23516/data/exid_dreamer_data \
    --train-locs 0 2 4 5 6 \
    --eval-locs 1 3 \
    --max-traj-per-loc 20 \
    --total-episodes 500 \
    --train-steps-per-collect 50 \
    --eval-interval 50 \
    --eval-episodes 10 \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_dreamer \
    --save-every 100 \
    --wandb-run-name "online_dreamer_v1"

echo "Done: $(date)"
