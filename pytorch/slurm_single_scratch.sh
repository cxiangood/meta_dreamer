#!/bin/bash
#SBATCH --job-name=single_scr
#SBATCH --comment="From-scratch DreamerV3 on single track: random init, high explore, 2000ep"
#SBATCH --partition=A800
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_scratch_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_scratch_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_single_scratch
mkdir -p "$OUTDIR"
rm -rf "$OUTDIR/videos"
mkdir -p "$OUTDIR/videos"

echo "=== From-scratch DreamerV3: single track, random init, explore_std=5.0 ==="
echo "Start: $(date)"

python3 -c "
import json
sel = {'6': [{'rid': 79, 'tid': 97, 'merge_idx': 59, 'loc_id': 6, 'density_bin': 'LOW', 'density': 5.0}]}
with open('$OUTDIR/single_track.json', 'w') as f:
    json.dump(sel, f)
"

python -u train_online_dreamer_v2.py \
    --from-scratch \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --train-locs 6 \
    --selection-file "$OUTDIR/single_track.json" \
    --total-episodes 2000 \
    --gt-warmup-per-track 10 \
    --curriculum-per-track 10 \
    --episodes-per-track 2000 \
    --explore-std 1.0 \
    --train-ratio 32 \
    --policy-steps 10 \
    --eval-interval 100 \
    --eval-episodes 1 \
    --eval-same-track \
    --batch-size 16 \
    --batch-length 50 \
    --seed 42 \
    --logdir "$OUTDIR" \
    --video-dir "$OUTDIR/videos" \
    --save-every 200 \
    --wandb-run-name single_scratch_2k

echo "Done: $(date)"
