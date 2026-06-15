#!/bin/bash
#SBATCH --job-name=df_sig_topo
#SBATCH --comment="DF+SIG+Topo: Topology-Guided WM with phase prediction auxiliary head (ramp/merge/main)"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_topo_p2_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_topo_p2_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== DF+SIG+Topo: Topology-Guided Decoder-Free World Model ==="
echo "  Phase prediction head: 3-class (ramp / merge / main)"
echo "  Frontend: strided conv (factor=2)"
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
    --use-phase-head \
    --phase-head-weight 1.0 \
    --merge-zone-frames 20 \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_topo_p2 \
    --bev-size 300 \
    --batch-size 16 \
    --total-steps 500000 \
    --sigreg-lambda 0.1 \
    --wandb-run-name df_sig_topo300_p2_s42 \
    --seed 42

echo "Done: $(date)"
