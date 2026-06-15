#!/bin/bash
#SBATCH --job-name=expJ_actbon
#SBATCH --comment="Exp J: action-dependent phase bonus in imagination (trainer.py)"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expJ_actbon_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expJ_actbon_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

WM_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph/checkpoint_step14000.pt
OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expJ_action_bonus
VIDEO_DIR="$OUTDIR/videos_train"
SEL_JSON="$OUTDIR/single_track.json"
mkdir -p "$OUTDIR" "$VIDEO_DIR"

echo "=== Exp J: action-dependent phase bonus in imagination ==="
echo "Changes from Exp I:"
echo "  trainer.py: imagination reward = base_reward + 0.3 × action_bonus"
echo "  Phase 0: +0.3*throttle+, -0.1*|steer| (ramp: accel + lane-keep)"
echo "  Phase 1: +0.2*|steer|, -0.1*|brake| (merge: steer + no-brake)"
echo "  Phase 2: +0.1*throttle+, -0.1*steer² (cruise + center steering)"
echo "  Branch: feat/action-bonus"
echo "WM: $WM_CKPT"
echo "Log: $OUTDIR"
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
    --no-residual-action \
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
    --bc-anchor-weight 0.05 \
    --gt-bc-weight 0.3 \
    --wandb-run-name expJ_action_bonus

echo "Done: $(date)"
