#!/bin/bash
#SBATCH --job-name=single_test
#SBATCH --comment="Single track test: P2 WM + pure online RL (no BC)"
#SBATCH --partition=A800
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_single_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_single_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

WM_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph/checkpoint_step14000.pt
OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_single_test
mkdir -p "$OUTDIR"
rm -rf "$OUTDIR/videos"
mkdir -p "$OUTDIR/videos"

echo "=== Single Track: P2 WM + pure online RL (no BC) ==="
echo "WM: df_ph/checkpoint_step14000.pt (phase_head 99% acc)"
echo "Actor: randomly initialized (no BC)"
echo "explore_std=5.0 | gt_warmup=5 | curriculum=5"
echo "Start: $(date)"

[ ! -f "$WM_CKPT" ] && echo "ERROR: WM not found" && exit 1

python3 -c "
import json
sel = {'0': [{'rid': 14, 'tid': 688, 'merge_idx': 408, 'loc_id': 0, 'density_bin': 'MID', 'density': 11.9}]}
with open('$OUTDIR/single_track.json', 'w') as f:
    json.dump(sel, f)
"

python -u train_online_dreamer_v2.py \
    --wm-ckpt "$WM_CKPT" \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --train-locs 0 \
    --selection-file "$OUTDIR/single_track.json" \
    --total-episodes 50 \
    --gt-warmup-per-track 5 \
    --curriculum-per-track 5 \
    --episodes-per-track 50 \
    --explore-std 5.0 \
    --train-ratio 32 \
    --policy-steps 10 \
    --eval-interval 25 \
    --eval-episodes 1 \
    --eval-same-track \
    --batch-size 16 \
    --batch-length 50 \
    --seed 42 \
    --logdir "$OUTDIR" \
    --video-dir "$OUTDIR/videos" \
    --save-every 50 \
    --wandb-run-name single_test_no_bc

echo "Done: $(date)"
