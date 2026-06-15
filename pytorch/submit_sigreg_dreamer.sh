#!/bin/bash
#SBATCH --job-name=sigreg_dreamer
#SBATCH --comment="SIGReg-Dreamer PyTorch Highway Merge"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64g
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/%x_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1504047409@qq.com

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "=========================================="

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

cd /share/home/u23516/code/meta_dreamer-main/pytorch || exit 1

# Activate conda environment
source /share/home/u23516/miniforge3/etc/profile.d/conda.sh
conda activate jepadrive

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")')"

LOGDIR="./logs/sigreg_dreamer_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# Quick test
echo ""
echo "=== Quick Test ==="
python main.py --mode test --bev-size 300 --sigreg-lambda 0.1

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Training ==="
    python main.py --mode train \
        --logdir "$LOGDIR" \
        --total-steps 1000000 \
        --bev-size 300 \
        --sigreg-lambda 0.1 \
        --seed 42 \
        --gpu 0
else
    echo "Quick test failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "End: $(date)"
echo "=========================================="
