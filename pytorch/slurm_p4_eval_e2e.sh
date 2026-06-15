#!/bin/bash
#SBATCH --job-name=p4_eval_e2e
#SBATCH --comment="P4 eval: E2E v2 (step2000) on loc1+loc3"
#SBATCH --partition=A800
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_e2e_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_e2e_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export SUMO_HOME="$CONDA_PREFIX/lib/python3.11/site-packages/sumo"

CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_e2e/checkpoint_latest.pt
OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_e2e
mkdir -p "$OUTDIR"

echo "=== P4 Eval: E2E v2 (step2000) ==="
echo "Checkpoint: $CKPT"
echo "Start: $(date)"

python -u eval_exid_phase4.py \
    --wm-ckpt "$CKPT" \
    --ac-ckpt "$CKPT" \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --loc 1 3 \
    --max-episodes 50 \
    --out "$OUTDIR/phase4_results.json" \
    --seed 42

echo "Done: $(date)"
echo "Results: $OUTDIR/phase4_results.json"
