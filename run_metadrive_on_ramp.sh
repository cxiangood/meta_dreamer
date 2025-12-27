#!/bin/bash

# MetaDrive On-Ramp 训练脚本（本地运行）

cd /share/home/u23516/code/meta_dreamer-main || exit 1

# 设置环境变量（可选）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_CPP_MIN_LOG_LEVEL=1

# 日志目录
LOGDIR="./dreamer/logs_metadrive_on_ramp/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# 运行训练
python3 dreamer/dreamerv3/main.py \
  --configs metadrive_on_ramp \
  --logdir "$LOGDIR" \
  --run.log_every 120 \
  --run.report_every 300 \
  --run.save_every 900 \
  --run.steps 1e6 \
  --batch_size 16 \
  --jax.platform cuda

