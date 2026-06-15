#!/bin/bash
# Sync wandb offline data from HPC to cloud
# Run periodically: watch -n 600 ./sync_wandb.sh
set -e

HPC="hpcAmd"
HPC_BASE="/share/home/u23516/code/meta_dreamer-main/pytorch/logs"
LOCAL_DIR="/tmp/wandb_sync"

echo "=== $(date) ==="

# Download wandb dirs from running training jobs
for job in df_sigreg_p2 df_sig_p3 df_sig_e2e df_sig_cnn_p3 df_sig_cnn_p2 df_barlow_p2 kl_p2 sigreg_p2 sigreg_deter_p2 dapo_p4_v2; do
    hpc_wandb="${HPC_BASE}/${job}/wandb"
    if ssh -o ConnectTimeout=5 "$HPC" "test -d $hpc_wandb" 2>/dev/null; then
        rsync -az --timeout=10 "$HPC:$hpc_wandb/" "$LOCAL_DIR/wandb/" 2>/dev/null && \
        echo "  Synced $job"
    fi
done

# Sync to wandb cloud
cd "$LOCAL_DIR"
for dir in wandb/run-*; do
    [ -d "$dir" ] && wandb sync "$dir" 2>&1 | grep -v "^$" | tail -1
done

# Also sync project dirs
for dir in wandb/df_* wandb/kl_* wandb/sigreg_* wandb/dapo_* wandb/p3_*; do
    if [ -d "$dir" ]; then
        wandb sync "$dir" 2>/dev/null && echo "  Synced $dir"
    fi
done

echo "Done."
