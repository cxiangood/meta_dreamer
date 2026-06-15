#!/bin/bash
#SBATCH --job-name=df_sig_jepa_traj
#SBATCH --comment="DF+SIG+JEPA+Traj+Speed: JEPA future-feature prediction + ego-centric trajectory prediction + speed head"
#SBATCH --partition=A800
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_cnn_jepa_traj_p2_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_cnn_jepa_traj_p2_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== DF+SIG+JEPA+Traj+Speed: dense self-supervised WM ==="
echo "JEPA: feature[t] + action[t] → feature[t+1] (3072-dim dynamics)"
echo "TrajHead: feature[t] → ego-centric future (Δx,Δy) for H=10 frames"
echo "Speed: feature[t] → ego speed (m/s, symlog)"
echo "This addresses the '3 scalars guide 3072 dims' problem with dense signals"
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
    --use-jepa \
    --jepa-weight 0.1 \
    --jepa-k 1 \
    --use-traj-head \
    --traj-head-weight 0.1 \
    --traj-horizon 10 \
    --use-speed-head \
    --speed-head-weight 0.1 \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_cnn_jepa_traj \
    --bev-size 300 \
    --batch-size 16 \
    --total-steps 500000 \
    --sigreg-lambda 0.1 \
    --log-every 50 \
    --wandb-run-name df_sig_cnn_jepa_traj_s42 \
    --seed 42

echo "Done: $(date)"
