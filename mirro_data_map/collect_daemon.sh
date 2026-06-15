#!/bin/bash
# 健壮的数据收集守护脚本
# - 跳过已完成的轨迹 (本地 mp4 或 npz 存在)
# - 失败不中断，记录到 failed.log
# - 每个 recording 完成后同步 HPC
# - 每 50 条报告进度
#
# 用法:
#   bash mirro_data_map/collect_daemon.sh          # 跑剩余训练集
#   bash mirro_data_map/collect_daemon.sh 4         # 只跑 loc 4
#   bash mirro_data_map/collect_daemon.sh --resume  # 从失败日志重试

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/collect_merge_data.py"
DATA_DIR="$SCRIPT_DIR/exid_dreamer_data"
CACHE="$SCRIPT_DIR/exid_merge_cache.json"
FAILED_LOG="$DATA_DIR/failed_$(date +%Y%m%d_%H%M%S).log"
HPC_DEST="hpcAmd:/share/home/u23516/data/exid_dreamer_data"
RSYNC_SSH_OPTS="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3"
HPC_HOST="${HPC_DEST%%:*}"
HPC_BASE="${HPC_DEST#*:}"
RSYNC_PARALLEL="${RSYNC_PARALLEL:-4}"

TRAIN_LOCS=(0 2 4 5 6)  # train first
TEST_LOCS=(1 3)          # test last

SUMO_HOME="${SUMO_HOME:-/Library/Frameworks/EclipseSUMO.framework/Versions/1.26.0/EclipseSUMO/share/sumo}"
export SUMO_HOME
export METADRIVE_HEADLESS=1

sync_rec_npz_parallel() {
    local rec_dir="$1"
    local loc_id="$2"
    local rec_id="$3"
    local remote_dir="$HPC_BASE/loc$loc_id/rec$(printf '%02d' "$rec_id")"
    local pids=()
    local running=0
    local fail=0

    if ! ssh "$HPC_HOST" "mkdir -p '$remote_dir'"; then
        echo "  rsync FAILED (mkdir remote dir), keeping npz locally"
        return 1
    fi

    shopt -s nullglob
    local npz_files=("$rec_dir"/*.npz)
    shopt -u nullglob
    if [ ${#npz_files[@]} -eq 0 ]; then
        return 0
    fi

    for f in "${npz_files[@]}"; do
        rsync -az --timeout=120 -e "$RSYNC_SSH_OPTS" \
            --partial --append \
            "$f" "$HPC_HOST:$remote_dir/" >> "$DATA_DIR/collect_output.log" 2>&1 &
        pids+=($!)
        running=$((running + 1))

        if [ "$running" -ge "$RSYNC_PARALLEL" ]; then
            for pid in "${pids[@]}"; do
                wait "$pid" || fail=1
            done
            pids=()
            running=0
        fi
    done

    if [ "$running" -gt 0 ]; then
        for pid in "${pids[@]}"; do
            wait "$pid" || fail=1
        done
    fi

    if [ "$fail" -eq 0 ]; then
        find "$rec_dir" -name "*.npz" -delete
        echo "  synced & cleaned (parallel=$RSYNC_PARALLEL)"
        return 0
    fi

    echo "  rsync FAILED (partial), keeping npz locally"
    return 1
}

if [ "$1" = "--resume" ]; then
    # 从最新的失败日志重试
    LATEST_FAIL=$(ls -t "$DATA_DIR"/failed_*.log 2>/dev/null | head -1)
    if [ -z "$LATEST_FAIL" ]; then
        echo "No failed log found"; exit 1
    fi
    echo "=== Retrying from $LATEST_FAIL ==="
    cat "$LATEST_FAIL"
    exit 0
fi

if [ -n "$1" ] && [ "$1" != "--resume" ]; then
    LOCS=("$1")
else
    LOCS=("${TRAIN_LOCS[@]}" "${TEST_LOCS[@]}")
fi

echo "=== 数据收集守护 $(date) ==="
echo "Locations: ${LOCS[*]}"
echo "Failed log: $FAILED_LOG"
echo ""

total_done=0
total_skip=0
total_fail=0
total_count=0

# Count total first
for loc in "${LOCS[@]}"; do
    n=$(python3 -c "
import json
with open('$CACHE') as f: c = json.load(f)
print(len(c.get('$loc', [])))
")
    total_count=$((total_count + n))
done
echo "待处理轨迹: $total_count"
echo ""

for loc in "${LOCS[@]}"; do
    echo "=============================="
    echo "Location $loc - $(date)"
    echo "=============================="

    # Read cache entries for this loc
    entries=$(python3 -c "
import json
with open('$CACHE') as f: c = json.load(f)
for e in c.get('$loc', []):
    print(e['rid'], e['tid'])
" "$CACHE")

    prev_rid=""

    while IFS=' ' read -r rid tid; do
        [ -z "$rid" ] && continue

        rec_dir="$DATA_DIR/rec$(printf '%02d' $rid)"
        npz_file="$rec_dir/track${tid}.npz"
        mp4_file="$rec_dir/track${tid}_bev.mp4"

        # Skip if already collected (npz or mp4 exists)
        if [ -f "$npz_file" ] || [ -f "$mp4_file" ]; then
            total_skip=$((total_skip + 1))
            continue
        fi

        mkdir -p "$rec_dir"

        # Run collection
        start_time=$(date +%s)
        if python3 "$SCRIPT" --recording "$rid" --track-id "$tid" --out-dir "$DATA_DIR" >> "$DATA_DIR/collect_output.log" 2>&1; then
            # Verify output exists
            if [ -f "$npz_file" ] || [ -f "$mp4_file" ]; then
                total_done=$((total_done + 1))
            else
                total_fail=$((total_fail + 1))
                echo "$(date +%H:%M:%S) FAILED (no output): rec$(printf '%02d' $rid) track$tid" >> "$FAILED_LOG"
            fi
        else
            total_fail=$((total_fail + 1))
            echo "$(date +%H:%M:%S) FAILED (exit $?): rec$(printf '%02d' $rid) track$tid" >> "$FAILED_LOG"
        fi

        elapsed=$(($(date +%s) - start_time))

        # Progress report every 50 or on failure
        processed=$((total_done + total_skip + total_fail))
        if [ $((processed % 50)) -eq 0 ] || [ $total_fail -gt 0 ]; then
            echo "  [$(date +%H:%M:%S)] 进度: $processed/$total_count | 完成: $total_done | 跳过: $total_skip | 失败: $total_fail | last: ${elapsed}s"
        fi

        # Sync when recording changes
        if [ "$rid" != "$prev_rid" ] && [ -n "$prev_rid" ]; then
            old_rec_dir="$DATA_DIR/rec$(printf '%02d' $prev_rid)"
            n_npz=$(find "$old_rec_dir" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
            if [ "$n_npz" -gt 0 ]; then
                echo "  syncing rec$(printf '%02d' $prev_rid) ($n_npz npz) → HPC loc$loc/ (parallel=$RSYNC_PARALLEL)..."
                sync_rec_npz_parallel "$old_rec_dir" "$loc" "$prev_rid"
            fi
        fi
        prev_rid="$rid"

    done <<< "$entries"

    # Sync last recording
    if [ -n "$prev_rid" ]; then
        last_rec_dir="$DATA_DIR/rec$(printf '%02d' $prev_rid)"
        n_npz=$(find "$last_rec_dir" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$n_npz" -gt 0 ]; then
            echo "  syncing rec$(printf '%02d' $prev_rid) ($n_npz npz) → HPC loc$loc/ (parallel=$RSYNC_PARALLEL)..."
            sync_rec_npz_parallel "$last_rec_dir" "$loc" "$prev_rid"
        fi
    fi

    echo "  Location $loc done: 完成=$total_done 跳过=$total_skip 失败=$total_fail"
done

echo ""
echo "=== 全部完成: $(date) ==="
echo "总计: $total_count | 完成: $total_done | 跳过: $total_skip | 失败: $total_fail"
if [ -f "$FAILED_LOG" ]; then
    echo "失败轨迹记录: $FAILED_LOG"
    cat "$FAILED_LOG"
fi