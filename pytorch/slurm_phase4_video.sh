#!/bin/bash
#SBATCH --job-name=p4_video
#SBATCH --comment="Phase4 video validation: DF+SIG on loc1 with BEV video export"
#SBATCH --partition=A800
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_video_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_video_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive

cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export SUMO_HOME="$CONDA_PREFIX/lib/python3.11/site-packages/sumo"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Phase 4: DF+SIG closed-loop on loc1 (video)"
echo "Start: $(date)"
echo "=========================================="

python -u eval_exid_phase4.py \
    --wm-ckpt /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2/checkpoint_latest.pt \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --loc 1 \
    --max-episodes 3 \
    --save-video \
    --video-dir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_video \
    --out /share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_video/phase4_results.json \
    --seed 42

echo "Done: $(date)"
echo "Videos in: /share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_video/"
ls -lh /share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_video/*.mp4 2>/dev/null
ls -lh /share/home/u23516/code/meta_dreamer-main/pytorch/logs/p4_video/*.json 2>/dev/null
