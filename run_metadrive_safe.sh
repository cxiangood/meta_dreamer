#!/bin/bash

# MetaDrive训练脚本，包含自动重启功能以避免OOM
# 使用更小的模型配置(size1m)和batch_size=8

cd /home/xiongxi/桌面/worldmodel_dreamerv3

# 激活正确的conda环境
conda activate python311

# 设置环境变量
export METADRIVE_FIXED_SEED=42

# 训练循环，每次运行固定时长后重启
while true; do
    echo "Starting DreamerV3 training at $(date)"

    # 运行训练，设置超时避免无限运行
    timeout 3600 python3 dreamer/dreamerv3/main.py --configs metadrive_lane_keeping --logdir ./dreamer/logs

    exit_code=$?
    echo "Training exited with code $exit_code at $(date)"

    if [ $exit_code -eq 124 ]; then
        echo "Training was terminated by timeout, restarting..."
    elif [ $exit_code -eq 137 ]; then
        echo "Training was killed (possibly OOM), restarting with fresh process..."
    else
        echo "Training completed or exited normally"
        break
    fi

    # 等待几秒钟再重启
    sleep 5
done