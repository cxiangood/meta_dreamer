#!/bin/bash
#SBATCH --job-name=expK0_smoke
#SBATCH --comment="ExpK0: crash detector smoke test, 80 episodes"
#SBATCH --partition=A800
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expK0_smoke_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expK0_smoke_%j.err

set -euo pipefail

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_MODE=offline
export WANDB_ENTITY="${WANDB_ENTITY:-jiojioxu-tongji-university}"

WM_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph/checkpoint_step14000.pt
OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expK0_crash_smoke
VIDEO_DIR="$OUTDIR/videos_train"
SEL_JSON="$OUTDIR/single_track.json"
mkdir -p "$OUTDIR" "$VIDEO_DIR"

echo "=== ExpK0: crash detector smoke test ==="
echo "Purpose: verify online crash detector learns before full ExpK."
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
    --total-episodes 80 \
    --gt-warmup-per-track 5 \
    --curriculum-per-track 5 \
    --episodes-per-track 80 \
    --explore-std 0.5 \
    --no-residual-action \
    --train-ratio 32 \
    --policy-steps 10 \
    --eval-interval 20 \
    --eval-episodes 2 \
    --eval-same-track \
    --batch-size 16 \
    --batch-length 50 \
    --seed 42 \
    --logdir "$OUTDIR" \
    --video-dir "$VIDEO_DIR" \
    --video-every 10 \
    --save-every 20 \
    --bc-anchor-weight 0.05 \
    --gt-bc-weight 0.3 \
    --wandb-run-name expK0_crash_smoke

echo "Done: $(date)"
