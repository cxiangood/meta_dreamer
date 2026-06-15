#!/bin/bash
#SBATCH --job-name=dapo_p4
#SBATCH --comment="P4 DAPO v1: K=8 online post-training with collision proxy"
#SBATCH --partition=A800
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/dapo_p4_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/dapo_p4_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline
export SUMO_HOME="$CONDA_PREFIX/lib/python3.11/site-packages/sumo"

echo "=== P4 DAPO v1 (K=8, collision proxy, online post-training) ==="
echo "Start: $(date)"

python -u train_dapo_p4.py \
    --wm-ckpt /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2/checkpoint_latest.pt \
    --ac-ckpt /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_p3/checkpoint_latest.pt \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --loc 1 3 \
    --max-episodes 200 \
    --k 8 \
    --collision-weight 5.0 \
    --lr 3e-5 \
    --seed 42 \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/dapo_p4

echo "Done: $(date)"
