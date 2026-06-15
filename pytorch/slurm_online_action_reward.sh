#!/bin/bash
#SBATCH --job-name=act_rwd
#SBATCH --comment="Online DreamerV3: phase+action_bonus imagination (no reward_head), advnorm off"
#SBATCH --partition=A800
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_act_rwd_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_act_rwd_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

# phase_head @ 99% (df_ph); BC actor init
WM_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph/checkpoint_step14000.pt
AC_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_vc_base/checkpoint_bc_best.pt
OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_act_rwd
mkdir -p "$OUTDIR"

echo "=== Online: phase + action_bonus imagination (see docs/analysis/online_reward_redesign.md) ==="
echo "WM: $WM_CKPT"
echo "AC: $AC_CKPT"
echo "OUT: $OUTDIR"
echo "Start: $(date)"

[ ! -f "$WM_CKPT" ] && echo "ERROR: WM not found: $WM_CKPT" && exit 1
[ ! -f "$AC_CKPT" ] && echo "ERROR: AC not found: $AC_CKPT" && exit 1

python -u train_online_dreamer_v2.py \
    --wm-ckpt "$WM_CKPT" \
    --ac-ckpt "$AC_CKPT" \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --train-locs 0 2 4 5 6 \
    --eval-locs 1 3 \
    --selection-file ./logs/exid_online_training_tracks.json \
    --total-episodes 1000 \
    --gt-warmup-per-track 3 \
    --curriculum-per-track 5 \
    --episodes-per-track 66 \
    --explore-std 1.0 \
    --train-ratio 32 \
    --policy-steps 10 \
    --eval-interval 50 \
    --eval-episodes 10 \
    --batch-size 16 \
    --batch-length 50 \
    --seed 42 \
    --logdir "$OUTDIR" \
    --video-dir "$OUTDIR/videos" \
    --save-every 50 \
    --wandb-run-name online_action_reward_v2

echo "Done: $(date)"
