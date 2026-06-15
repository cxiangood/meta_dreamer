#!/bin/bash
#SBATCH --partition=L40
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=64G
#SBATCH --job-name=online_no_wp
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_no_wp_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_no_wp_%j.err

module load sumo/1.20
source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
export PYTHONPATH="$SUMO_HOME/share/sumo/tools:$PYTHONPATH"
export SUMO_HOME="/share/apps/sumo-1.20"
export WANDB_API_KEY=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
export WANDB_ENTITY=jiojioxu-tongji-university
export WANDB_MODE=offline
export METADRIVE_HEADLESS=1
cd /share/home/u23516/code/meta_dreamer-main/pytorch
mkdir -p logs/online_v2 logs/online_videos
exec python -u train_online_dreamer_v2.py \
    --wm-ckpt logs/df_sig_cnn_jepa_traj/checkpoint_step2000.pt \
    --ac-ckpt logs/bc_jepa_traj/checkpoint_bc_best.pt \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --npz-dir /share/home/u23516/data/exid_dreamer_data \
    --train-locs 0 2 4 5 6 \
    --eval-locs 1 3 \
    --max-traj-per-loc 15 \
    --selection-file /share/home/u23516/code/meta_dreamer-main/mirro_data_map/exid_online_selection.json \
    --total-episodes 150 \
    --batch-size 8 \
    --train-ratio 32 \
    --policy-steps 50 \
    --eval-interval 30 \
    --eval-episodes 5 \
    --bc-weight 0.5 \
    --video-dir logs/online_videos \
    --logdir logs/online_v2 \
    --save-every 50 \
    --wandb-run-name online_dreamer_v2_no_wp
