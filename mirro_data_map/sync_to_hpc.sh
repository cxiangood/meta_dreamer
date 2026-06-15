#!/bin/bash
# 将 exiD Dreamer 训练数据同步到 HPC（排除视频）
# 用法:
#   ./sync_to_hpc.sh           # 增量同步
#   ./sync_to_hpc.sh --dry-run # 预览
#   ./sync_to_hpc.sh --all     # 全量同步

set -e

HPC_HOST="hpcAmd"
HPC_DIR="/share/home/u23516/code/meta_dreamer-main/pytorch/data/exid_dreamer_data"
LOCAL_DIR="$(dirname "$0")/exid_dreamer_data"

DRY_RUN=""
UPDATE="--update"

if [ "$1" = "--dry-run" ]; then
    DRY_RUN="--dry-run"
elif [ "$1" = "--all" ]; then
    UPDATE=""
fi

echo "Syncing: $LOCAL_DIR/ → $HPC_HOST:$HPC_DIR/"
echo "Exclude: *.mp4"
echo ""

rsync -avz $UPDATE $DRY_RUN --exclude="*.mp4" "$LOCAL_DIR/" "$HPC_HOST:$HPC_DIR/"

echo ""
echo "Done!"
