#!/bin/bash
#SBATCH --job-name=online_v2
#SBATCH --comment="Online DreamerV3: P2 WM + pure online RL (no BC, phase_head reward)"
#SBATCH --partition=A800
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_v2_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_v2_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_v2
mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_videos

module load sumo/1.20
source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export PYTHONPATH="$SUMO_HOME/share/sumo/tools:$PYTHONPATH"
export SUMO_HOME="/share/apps/sumo-1.20"

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline
export METADRIVE_HEADLESS=1

echo "=== Online DreamerV3: P2 WM + pure online RL (no BC) ==="
echo "WM: df_ph/checkpoint_step14000.pt (phase_head 99% acc)"
echo "Actor: randomly initialized, explore_std=5.0"
echo "Loop: interact 10 steps → train (ratio=32) → repeat"
echo "Reward: phase_head imagination (no reward_head)"
echo "Per-track: GT_warmup=5 curriculum=5 → pure RL"
echo "Start: $(date)"

python -u train_online_dreamer_v2.py \
    --wm-ckpt /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph/checkpoint_step14000.pt \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --npz-dir /share/home/u23516/data/exid_dreamer_data \
    --train-locs 0 2 4 5 6 \
    --eval-locs 1 3 \
    --max-traj-per-loc 15 \
    --selection-file /share/home/u23516/code/meta_dreamer-main/mirro_data_map/exid_online_selection.json \
    --total-episodes 200 \
    --train-ratio 32 \
    --policy-steps 10 \
    --eval-interval 50 \
    --eval-episodes 5 \
    --gt-warmup-per-track 5 \
    --curriculum-per-track 5 \
    --explore-std 5.0 \
    --batch-size 16 \
    --batch-length 50 \
    --video-dir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_videos \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_v2 \
    --save-every 50 \
    --seed 42 \
    --wandb-run-name "online_dreamer_v2_no_bc"

echo "Done: $(date)"
