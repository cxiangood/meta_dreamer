#!/bin/bash
#SBATCH --job-name=p4_eval_cnn_bc
#SBATCH --comment="P4 eval: CNN P2 step4000 + BC pretrain on loc1+loc3"
#SBATCH --partition=A800
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_cnn_bc_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_cnn_bc_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export SUMO_HOME="$CONDA_PREFIX/lib/python3.11/site-packages/sumo"

# WM checkpoint (CNN P2, frozen)
WM_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_cnn_p2/checkpoint_step4000.pt

# BC checkpoint (actor trained to mimic GT in WM feature space)
# Use bc_best if available, else latest
BC_DIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_cnn
if [ -f "$BC_DIR/checkpoint_bc_best.pt" ]; then
    BC_CKPT="$BC_DIR/checkpoint_bc_best.pt"
elif [ -f "$BC_DIR/checkpoint_bc_final.pt" ]; then
    BC_CKPT="$BC_DIR/checkpoint_bc_final.pt"
else
    BC_CKPT="$BC_DIR/checkpoint_latest.pt"
fi

OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_cnn_bc
mkdir -p "$OUTDIR"

echo "=== P4 Eval: CNN WM + BC Actor ==="
echo "WM checkpoint: $WM_CKPT"
echo "AC checkpoint: $BC_CKPT"
echo "Start: $(date)"

if [ ! -f "$BC_CKPT" ]; then
    echo "ERROR: BC checkpoint not found at $BC_CKPT"
    echo "Run slurm_bc_cnn_pretrain.sh first!"
    exit 1
fi

python -u eval_exid_phase4.py \
    --wm-ckpt "$WM_CKPT" \
    --ac-ckpt "$BC_CKPT" \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --loc 1 3 \
    --max-episodes 100 \
    --out "$OUTDIR/phase4_results.json" \
    --seed 42

echo "Done: $(date)"
echo "Results: $OUTDIR/phase4_results.json"
