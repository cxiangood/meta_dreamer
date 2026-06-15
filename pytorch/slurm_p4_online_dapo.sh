#!/bin/bash
#SBATCH --job-name=p4_online_dapo
#SBATCH --comment="P4 Online DAPO: K=8 real env with collision penalty"
#SBATCH --partition=A800
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_online_dapo_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_online_dapo_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline
export SUMO_HOME="$CONDA_PREFIX/lib/python3.11/site-packages/sumo"

echo "=== P4 Online DAPO (K=8, real env, custom collision penalty) ==="
echo "Start: $(date)"

python -u train_p4_online.py     --wm-ckpt /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2/checkpoint_step4000.pt     --ac-ckpt /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_p3/checkpoint_step4000.pt     --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data     --loc 1 3     --max-episodes 200     --k 8     --collision-weight 50.0     --success-bonus 20.0     --lr 1e-5     --eval-every 10     --seed 42     --scratch --init-log-std 0.0 --steer-scale 0.5 --record-video --record-first-n 5 --exploration-noise 0.0 --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_online_dapo

echo "Done: $(date)"
