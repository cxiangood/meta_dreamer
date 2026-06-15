#!/bin/bash
#SBATCH -J exid_loc
#SBATCH -p L40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 7
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -o logs/exid_loc_%x_loc%j.out
#SBATCH -e logs/exid_loc_%x_loc%j.err

if [ $# -lt 1 ]; then
    echo "Usage: sbatch slurm_collect_loc.sh <loc_id> [capture_every]"
    exit 1
fi

LOC_ID="$1"
CAPTURE_EVERY="${2:-1}"

# ===== 环境初始化 =====
source /share/home/u23516/miniforge3/etc/profile.d/conda.sh
conda activate metadrive

module load sumo/1.20
export SUMO_HOME="${SUMO_HOME}/share/sumo"

# ===== 路径 ====
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u23516/code/meta_dreamer-main}"
DATASET_ROOT="${EXID_DATASET_DIR:-/share/home/u23516/data/exiD-dataset-v2.1}"
OUT_DIR="${OUT_DIR:-/share/home/u23516/data/exid_dreamer_data}/loc${LOC_ID}"

cd "$PROJECT_ROOT"
mkdir -p logs "$OUT_DIR"

export EXID_DATASET_DIR="$DATASET_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export METADRIVE_HEADLESS=1
export MPLCONFIGDIR="/tmp/mpl_${USER}"
mkdir -p "$MPLCONFIGDIR"

# ===== Headless 渲染配置 =====
if ! pgrep -x Xvfb > /dev/null 2>&1; then
    XVFB_CMD=$(find /usr -name Xvfb -type f 2>/dev/null | head -1)
    if [ -n "$XVFB_CMD" ]; then
        XVFB_DISPLAY=$(((RANDOM % 100) + 100))
        "$XVFB_CMD" ":$XVFB_DISPLAY" -screen 0 1024x768x24 -ac +extension GLX +render &
        XVFB_PID=$!
        export DISPLAY=":$XVFB_DISPLAY"
        trap "kill $XVFB_PID 2>/dev/null" EXIT
        echo "Xvfb started on DISPLAY=:$XVFB_DISPLAY (pid=$XVFB_PID)"
    else
        export DISPLAY=":0"
    fi
fi

CACHE="$PROJECT_ROOT/mirro_data_map/exid_merge_cache.json"

echo "==== exiD collect (loc=${LOC_ID}) ===="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "EXID_DATASET_DIR=$EXID_DATASET_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "SUMO_HOME=$SUMO_HOME"
echo "DISPLAY=${DISPLAY:-not set}"
echo "CAPTURE_EVERY=$CAPTURE_EVERY"
date

# 从缓存读取该 location 的轨迹列表
if [ ! -f "$CACHE" ]; then
    echo "ERROR: cache not found: $CACHE"
    exit 1
fi

ENTRIES=$(python3 -c "
import json
with open('$CACHE') as f:
    data = json.load(f)
for entry in data.get('$LOC_ID', []):
    print(entry['rid'], entry['tid'])
")

N_TOTAL=$(echo "$ENTRIES" | wc -l | tr -d ' ')
echo "Total trajectories: $N_TOTAL"

done_count=0
skip_count=0
fail_count=0

while IFS=' ' read -r rid tid; do
    [ -z "$rid" ] && continue

    rec_dir="$OUT_DIR/rec$(printf '%02d' $rid)"
    npz_file="$rec_dir/track${tid}.npz"

    if [ -f "$npz_file" ]; then
        skip_count=$((skip_count + 1))
        continue
    fi

    mkdir -p "$rec_dir"

    # 每条轨迹独立 Python 进程，避免 Pand3D/全局状态残留
    if python3 mirro_data_map/collect_merge_data.py \
        --recording "$rid" \
        --track-id "$tid" \
        --capture-every "$CAPTURE_EVERY" \
        --out-dir "$OUT_DIR" 2>&1 | grep -q "saved"; then
        done_count=$((done_count + 1))
    else
        fail_count=$((fail_count + 1))
        echo "  FAILED: rec$rid track$tid @ $(date)"
    fi

    echo "  [loc${LOC_ID}] done=$done_count skip=$skip_count fail=$fail_count / $N_TOTAL"

done <<< "$ENTRIES"

echo ""
echo "==== done loc=${LOC_ID} ===="
echo "Total: $N_TOTAL | done=$done_count skip=$skip_count fail=$fail_count"
date
