#!/bin/bash
#SBATCH --job-name=expG_pure
#SBATCH --comment="Exp G: pure online DreamerV3 from scratch (no pretrained WM, 1w env steps)"
#SBATCH --partition=A800
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expG_pure_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expG_pure_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/expG_pure_online
VIDEO_DIR="$OUTDIR/videos_train"
SEL_JSON="$OUTDIR/single_track.json"
mkdir -p "$OUTDIR" "$VIDEO_DIR"

echo "=== Exp G: pure online DreamerV3 from scratch (no pretrained WM) ==="
echo "Scene: loc6_rec79_t97 (single track, same as E/F)"
echo "WM: from scratch (random init, train online with AC)"
echo "Target: ~50 episodes ≈ 1w env steps"
echo "Log/checkpoints: $OUTDIR"
echo "Start: $(date)"

python3 -c "
import json
sel = {'6': [{'rid': 79, 'tid': 97, 'merge_idx': 59, 'loc_id': 6, 'density_bin': 'LOW', 'density': 5.0}]}
with open('$SEL_JSON', 'w') as f:
    json.dump(sel, f)
"

python -u train_online_dreamer_v2.py \
    --from-scratch \
    --actor-init gt_buffer \
    --gt-actor-init-steps 200 \
    --success-sample-ratio 0.9 \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --train-locs 6 \
    --selection-file "$SEL_JSON" \
    --total-episodes 50 \
    --gt-warmup-per-track 5 \
    --curriculum-per-track 5 \
    --episodes-per-track 50 \
    --explore-std 0.5 \
    --no-residual-action \
    --train-ratio 32 \
    --policy-steps 10 \
    --eval-interval 10 \
    --eval-episodes 1 \
    --eval-same-track \
    --batch-size 16 \
    --batch-length 50 \
    --seed 42 \
    --logdir "$OUTDIR" \
    --video-dir "$VIDEO_DIR" \
    --video-every 5 \
    --save-every 50 \
    --bc-anchor-weight 0.05 \
    --gt-bc-weight 0.3 \
    --wandb-run-name expG_pure_online

echo "Done: $(date)"
