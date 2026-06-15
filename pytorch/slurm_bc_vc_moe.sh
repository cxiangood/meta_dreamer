#!/bin/bash
#SBATCH --job-name=bc_vc_moe
#SBATCH --comment="BC 8000 MoE-Phase on Veh+Curv WM (step6000)."
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_vc_moe_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_vc_moe_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

echo "=== BC 8000 MoE: Veh+Curv WM step6000 ==="
echo "Start: $(date)"

python -u main.py \
    --phase bc \
    --resume /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_traj_vc/checkpoint_step6000.pt \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_vc_moe \
    --bev-size 300 \
    --bev-downsample cnn \
    --cnn-factor 2 \
    --batch-size 16 \
    --bc-use-moe \
    --total-steps 8000 \
    --seed 42 \
    --wandb-run-name bc_vc_moe_s42

echo "Done: $(date)"
