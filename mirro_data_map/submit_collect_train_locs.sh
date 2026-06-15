#!/bin/bash
# 提交训练集 loc 的 Slurm 任务（可按需调整分区/资源）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/slurm_collect_loc.sh"

if [ ! -f "$SBATCH_SCRIPT" ]; then
    echo "Missing $SBATCH_SCRIPT"
    exit 1
fi

CAPTURE_EVERY="${CAPTURE_EVERY:-1}"
LOCS=("$@")
if [ ${#LOCS[@]} -eq 0 ]; then
    LOCS=(0 2 4 5 6)
fi

echo "Submitting loc jobs: ${LOCS[*]}"
echo "CAPTURE_EVERY=$CAPTURE_EVERY"

for loc in "${LOCS[@]}"; do
    echo "---- submit loc $loc ----"
    sbatch "$SBATCH_SCRIPT" "$loc" "$CAPTURE_EVERY"
done
