#!/bin/bash
#SBATCH --job-name=df_sig_p4
#SBATCH --comment="Phase4 online finetuning: MetaDrive procedural env"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_p4_online_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_p4_online_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive

cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export SUMO_HOME="$CONDA_PREFIX/lib/python3.11/site-packages/sumo"
export METADRIVE_HEADLESS=1

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Phase 4: Online finetuning (df_sigreg_p2 step2000 WM + random AC)"
echo "Start: $(date)"
echo "=========================================="

python -u main.py     --phase phase4     --reg sigreg     --sigreg-target deter+logits     --sigreg-lambda 0.1     --use-decoder False     --barlow-lambda 0.005     --barlow-k 1     --resume /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sigreg_p2/checkpoint_latest.pt     --logdir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_sig_p4_online     --bev-size 64     --batch-size 16     --total-steps 50000     --wandb-run-name df_sig_p4_online_s42     --seed 42

echo "Done: $(date)"
