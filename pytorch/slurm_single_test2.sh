#!/bin/bash
#SBATCH --job-name=single_1k
#SBATCH --comment="Track2 1000ep: alive_bonus + BC_reg + reward_head+phase_head"
#SBATCH --partition=A800
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_single2_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_single2_%j.err

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
rm -rf "$OUTDIR/videos2"
mkdir -p "$OUTDIR/videos2"

echo "=== loc6_rec79_t97 merge_idx=59 + action logging ==="
echo "Start: $(date)"

python3 -c "
import json
sel = {'6': [{'rid': 79, 'tid': 97, 'merge_idx': 59, 'loc_id': 6, 'density_bin': 'LOW', 'density': 5.0}]}
with open('$OUTDIR/single_track2.json', 'w') as f:
    json.dump(sel, f)
"

python -u train_online_dreamer_v2.py \
    --wm-ckpt "$WM_CKPT" \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --train-locs 6 \
    --selection-file "$OUTDIR/single_track2.json" \
    --total-episodes 1000 \
    --gt-warmup-per-track 5 \
    --curriculum-per-track 5 \
    --episodes-per-track 1000 \
    --explore-std 0.5 \
    --train-ratio 32 \
    --policy-steps 10 \
    --eval-interval 50 \
    --eval-episodes 1 \
    --eval-same-track \
    --batch-size 16 \
    --batch-length 50 \
    --seed 42 \
    --logdir "$OUTDIR" \
    --video-dir "$OUTDIR/videos2" \
    --save-every 100 \
    --wandb-run-name single_1k_alive_bc

echo "Done: $(date)"
