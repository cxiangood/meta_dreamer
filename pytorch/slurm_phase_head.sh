#!/bin/bash
#SBATCH --job-name=df_ph
#SBATCH --comment="Traj+Veh+Curv+PhaseHead. geometric merge endpoint labels."
#SBATCH --partition=A800
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== PhaseHead + Traj+Veh+Curv, sigreg=deter+logits ==="
echo "Geometric merge endpoint labels"
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
    --use-traj-head \
    --traj-head-weight 0.1 \
    --traj-horizon 125 \
    --use-vehicle-head \
    --vehicle-head-weight 0.1 \
    --n-surrounding-vehicles 5 \
    --use-curvature-head \
    --curvature-head-weight 0.15 \
    --use-phase-head \
    --phase-head-weight 1.0 \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph \
    --bev-size 300 \
    --batch-size 16 \
    --total-steps 500000 \
    --sigreg-lambda 0.1 \
    --log-every 50 \
    --wandb-run-name df_ph_geo_s42 \
    --seed 42

echo "Done: $(date)"
