#!/bin/bash
# 补传本地未同步到HPC的npz文件
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/exid_dreamer_data"
HPC_DEST="hpcAmd:/share/home/u23516/data/exid_dreamer_data"

# 遍历所有rec目录，找npz文件并同步
for rec_dir in "$DATA_DIR"/rec*; do
    [ -d "$rec_dir" ] || continue
    # 提取rec编号（如 rec00 → 00）
    rec_num=$(basename "$rec_dir" | sed 's/rec//')
    # 遍历该目录下的loc（如果分不清loc，可先同步到临时目录，再在HPC端整理）
    for loc in {0..6}; do
        # 同步npz文件（--append确保断点续传，不会重复传）
        rsync -av --partial --append \
            --include="*.npz" --exclude="*" \
            "$rec_dir/" \
            "$HPC_DEST/loc$loc/rec$rec_num/" 2>&1
        
        # 同步成功后删除本地npz（可选，和原脚本逻辑一致）
        if [ $? -eq 0 ]; then
            find "$rec_dir" -name "*.npz" -delete
            echo "✅ 已同步并清理: $rec_dir (loc$loc)"
        else
            echo "❌ 同步失败: $rec_dir (loc$loc)"
        fi
    done
done

echo "📌 历史数据补传完成！"