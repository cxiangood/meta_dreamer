#!/bin/bash
#SBATCH --job-name=p4_eval_p3
#SBATCH --comment="Phase4 evaluation: test P3 AC on exiD val set (loc1+loc3)"
#SBATCH --partition=A800
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_p3_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_p3_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive

cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export SUMO_HOME="$CONDA_PREFIX/lib/python3.11/site-packages/sumo"

P3_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_p3/checkpoint_latest.pt

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Phase 4 eval: P3 AC on loc1+loc3 val set"
echo "P3 checkpoint: $P3_CKPT"
echo "Start: $(date)"
echo "=========================================="

python -u eval_exid_phase4.py \
    --wm-ckpt "$P3_CKPT" \
    --ac-ckpt "$P3_CKPT" \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --loc 1 3 \
    --max-episodes 50 \
    --out /share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_p3/phase4_results.json \
    --seed 42

echo "Done: $(date)"
echo "Results in: /share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_eval_p3/"
