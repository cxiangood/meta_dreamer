#!/bin/bash
# 批量收集训练集数据 (Location 0,2,4,5,6)
# 从 exid_merge_cache.json 读取轨迹列表，避免重复筛选
# 每条轨迹独立 Python 进程，避免 Panda3D 残留
#
# 用法:
#   ./collect_all_train.sh           # 全部训练 location
#   ./collect_all_train.sh 5         # 只跑 Location 5
#   ./collect_all_train.sh --sync    # 只同步已有数据到 HPC（不收集）
#   ./collect_all_train.sh --list    # 列出待收集轨迹统计

set -e

SCRIPT_DIR="$(dirname "$0")"
SCRIPT="$SCRIPT_DIR/collect_merge_data.py"
DATA_DIR="$SCRIPT_DIR/exid_dreamer_data"
CACHE="$SCRIPT_DIR/exid_merge_cache.json"
HPC_DEST="hpcAmd:/share/home/u23516/data/exid_dreamer_data"

TRAIN_LOCS=(0 2 4 5 6)

sync_to_hpc() {
    local rec_dir="$1"
    local loc_id="$2"
    local rec_name="$(basename "$rec_dir")"
    echo "  syncing $rec_name npz → HPC loc$loc_id/..."
    rsync -avz --include="*.npz" --exclude="*" "$rec_dir/" "$HPC_DEST/loc$loc_id/$rec_name/"
    if [ $? -eq 0 ]; then
        find "$rec_dir" -name "*.npz" -delete
        local n_mp4=$(ls "$rec_dir"/*.mp4 2>/dev/null | wc -l)
        echo "  synced + cleaned, $n_mp4 videos kept locally"
    else
        echo "  rsync FAILED, keeping local npz"
    fi
}

if [ "$1" = "--sync" ]; then
    echo "=== 同步已有数据到 HPC ==="
    for rec_dir in "$DATA_DIR"/rec*/; do
        [ -d "$rec_dir" ] || continue
        rec_name="$(basename "$rec_dir")"
        rec_num="${rec_name#rec}"
        # 从 cache 查 loc_id
        loc=$(python3 -c "
import json
with open('$CACHE') as f:
    c = json.load(f)
for lid, entries in c.items():
    if any(e['rid'] == int('$rec_num') for e in entries):
        print(lid); break
" 2>/dev/null || echo "?")
        sync_to_hpc "$rec_dir" "$loc"
    done
    echo "Done!"
    exit 0
fi

if [ ! -f "$CACHE" ]; then
    echo "ERROR: 缓存文件不存在: $CACHE"
    echo "请先运行: python3 mirro_data_map/plot_merge_scenes.py --build-cache"
    exit 1
fi

# 从缓存读取轨迹列表: 输出 "rid tid" 每行
read_cache() {
    local loc=$1
    python3 -c "
import json
with open('$CACHE') as f:
    data = json.load(f)
for entry in data.get('$loc', []):
    print(entry['rid'], entry['tid'])
"
}

if [ "$1" = "--list" ]; then
    echo "=== 训练集轨迹统计 (from cache) ==="
    total=0
    for loc in "${TRAIN_LOCS[@]}"; do
        count=$(read_cache $loc | wc -l | tr -d ' ')
        total=$((total + count))
        printf "  Location %s: %s 条轨迹\n" "$loc" "$count"
    done
    echo "  总计: $total"
    exit 0
fi

if [ -n "$1" ]; then
    LOCS=("$1")
else
    LOCS=("${TRAIN_LOCS[@]}")
fi

echo "=== 训练集数据收集 (from cache) ==="
echo "Cache: $CACHE"
echo "Locations: ${LOCS[*]}"
echo "开始时间: $(date)"
echo ""

total=0
done=0
skipped=0
failed=0

for loc in "${LOCS[@]}"; do
    echo ""
    echo "=============================="
    echo "Location $loc - $(date)"
    echo "=============================="

    entries=$(read_cache $loc)
    n_total=$(echo "$entries" | wc -l | tr -d ' ')
    echo "  $n_total trajectories from cache"
    total=$((total + n_total))

    prev_rid=""
    while IFS=' ' read -r rid tid; do
        rec_dir="$DATA_DIR/rec$(printf '%02d' $rid)"
        npz_file="$rec_dir/track${tid}.npz"

        if [ -f "$npz_file" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        mkdir -p "$rec_dir"

        if python3 "$SCRIPT" --recording "$rid" --track-id "$tid" --out-dir "$DATA_DIR" 2>&1 | grep -q "saved"; then
            done=$((done + 1))
        else
            failed=$((failed + 1))
            echo "    FAILED: rec$rid track$tid"
        fi

        # 每个 recording 结束时同步
        if [ "$rid" != "$prev_rid" ] && [ -n "$prev_rid" ]; then
            old_rec_dir="$DATA_DIR/rec$(printf '%02d' $prev_rid)"
            sync_to_hpc "$old_rec_dir" "$loc"
        fi
        prev_rid="$rid"

    done <<< "$entries"

    # 同步最后一个 recording
    if [ -n "$prev_rid" ]; then
        sync_to_hpc "$DATA_DIR/rec$(printf '%02d' $prev_rid)" "$loc"
    fi

    echo "  Location $loc: done=$done skipped=$skipped failed=$failed"
done

echo ""
echo "=== 全部完成: $(date) ==="
echo "总计: $total 条 (完成: $done, 跳过: $skipped, 失败: $failed)"
echo ""
echo "文件统计:"
find "$DATA_DIR" -name "*.mp4" | wc -l | xargs echo "  .mp4 (视频):"
find "$DATA_DIR" -name "*.npz" | wc -l | xargs echo "  .npz (残留):"
du -sh "$DATA_DIR" | awk '{print "  本地总大小: " $1}'
