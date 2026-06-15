#!/bin/bash
#SBATCH --job-name=expF_rfwm
#SBATCH --comment="Exp F: reward_head + frozen WM + residual actions"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expF_rfwm_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expF_rfwm_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

WM_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph/checkpoint_step14000.pt
OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expF_rew_frozen_wm
VIDEO_DIR="$OUTDIR/videos_train"
SEL_JSON="$OUTDIR/single_track.json"
mkdir -p "$OUTDIR" "$VIDEO_DIR"

echo "=== Exp F: reward_head + frozen WM + residual actions ==="
echo "WM: $WM_CKPT (FROZEN)"
echo "Log/checkpoints: $OUTDIR"
echo "Start: $(date)"

[ ! -f "$WM_CKPT" ] && echo "ERROR: WM not found" && exit 1

python3 -c "
import json
sel = {'6': [{'rid': 79, 'tid': 97, 'merge_idx': 59, 'loc_id': 6, 'density_bin': 'LOW', 'density': 5.0}]}
with open('$SEL_JSON', 'w') as f:
    json.dump(sel, f)
"

python -u train_online_dreamer_v2.py \
    --wm-ckpt "$WM_CKPT" \
    --actor-init gt_buffer \
    --gt-actor-init-steps 200 \
    --failure-reflection \
    --success-sample-ratio 0.9 \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --train-locs 6 \
    --selection-file "$SEL_JSON" \
    --total-episodes 500 \
    --gt-warmup-per-track 5 \
    --curriculum-per-track 5 \
    --episodes-per-track 500 \
    --explore-std 0.5 \
    --freeze-wm \
    --train-ratio 32 \
    --policy-steps 10 \
    --eval-interval 25 \
    --eval-episodes 3 \
    --eval-same-track \
    --batch-size 16 \
    --batch-length 50 \
    --seed 42 \
    --logdir "$OUTDIR" \
    --video-dir "$VIDEO_DIR" \
    --video-every 10 \
    --save-every 50 \
    --wandb-run-name expF_rew_frozen_wm

echo "Done: $(date)"
