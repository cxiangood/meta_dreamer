#!/bin/bash
#SBATCH --job-name=p4eval_direct_vc
#SBATCH --comment="P4 Eval: Veh+Curv WM direct (no actor, random actions)"
#SBATCH --partition=A800
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4eval_direct_vc_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4eval_direct_vc_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline
export SUMO_HOME="$CONDA_PREFIX/lib/python3.11/site-packages/sumo"

WM_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_traj_vc/checkpoint_step6000.pt
OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4eval_direct_vc
mkdir -p "$OUTDIR"

echo "=== P4 Eval: Veh+Curv WM direct (random actor) ==="
echo "WM: $WM_CKPT"
echo "Start: $(date)"

python -u eval_exid_phase4.py \
    --wm-ckpt "$WM_CKPT" \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --loc 1 3 \
    --max-episodes 100 \
    --out "$OUTDIR/phase4_results.json" \
    --seed 42

echo "Done: $(date)"
