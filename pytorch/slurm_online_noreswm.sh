#!/bin/bash
#SBATCH --job-name=norwm_1t
#SBATCH --comment="NoResWM (no residual): loc6_rec79_t97, 1000ep, action_bonus imagination"
#SBATCH --partition=A800
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_noreswm_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_noreswm_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

export WANDB_API_KEY="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"
export WANDB_ENTITY="jiojioxu-tongji-university"
export WANDB_MODE=offline

WM_CKPT=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph/checkpoint_step14000.pt
OUTDIR=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_noreswm
SEL_JSON="$OUTDIR/single_track.json"
mkdir -p "$OUTDIR/videos"

echo "=== NoResWM (residual actions OFF): loc6_rec79_t97 merge_idx=59, 1000 episodes ==="
echo "WM: $WM_CKPT (offline P2, frozen phase_head)"
echo "Actor: residual_action=False (action = tanh(outscale*Δ), standard DreamerV3)"
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
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --train-locs 6 \
    --selection-file "$SEL_JSON" \
    --total-episodes 1000 \
    --gt-warmup-per-track 5 \
    --curriculum-per-track 5 \
    --episodes-per-track 1000 \
    --wm-warmup-episodes 25 \
    --explore-std 0.5 \
    --train-ratio 32 \
    --policy-steps 10 \
    --eval-interval 25 \
    --eval-episodes 3 \
    --eval-same-track \
    --batch-size 16 \
    --batch-length 50 \
    --seed 42 \
    --logdir "$OUTDIR" \
    --video-dir "$OUTDIR/videos" \
    --video-every 20 \
    --save-every 50 \
    --save-gt-video \
    --no-residual-action \
    --wandb-run-name online_noreswm

echo "Done: $(date)"
