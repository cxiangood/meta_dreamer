#!/bin/bash

# 同步Windows上的修改到WSL2
echo "开始同步文件到WSL2..."

# 定义源路径（Windows）和目标路径（WSL2）
WINDOWS_BASE="/mnt/d/学习/智能驾驶训练集生成/Prj_worldmoudle/dreamerv3-main/dreamerv3-main"
WSL_BASE="$HOME/smart_driving/Prj_worldmoudle/dreamerv3-main/dreamerv3-main"

# 同步关键文件
echo "同步 MetaDrive 环境文件..."
cp "$WINDOWS_BASE/embodied/envs/metadrive_lane_keeping.py" "$WSL_BASE/embodied/envs/"

echo "同步测试文件..."
cp "$WINDOWS_BASE/test_metadrive_env.py" "$WSL_BASE/"

echo "同步配置文件..."
cp "$WINDOWS_BASE/dreamerv3/configs.yaml" "$WSL_BASE/dreamerv3/"

echo "同步主程序文件..."
cp "$WINDOWS_BASE/dreamerv3/main.py" "$WSL_BASE/dreamerv3/"

echo "文件同步完成！"
echo "现在可以在WSL2中运行测试："
echo "  cd ~/smart_driving/Prj_worldmoudle/dreamerv3-main/dreamerv3-main"
echo "  python3 test_metadrive_env.py"