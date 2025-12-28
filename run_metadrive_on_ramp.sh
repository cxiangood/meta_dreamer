#!/bin/bash

# MetaDrive On-Ramp 训练脚本（本地运行）
# 匹配之前的运行方式：cd dreamerv3-main/dreamerv3-main && python dreamerv3/main.py --configs metadrive_on_ramp

# 设置环境变量（可选）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # 降低内存使用
export TF_CPP_MIN_LOG_LEVEL=1

# 切换到 dreamer 目录（相当于之前的 dreamerv3-main/dreamerv3-main）
cd "$(dirname "$0")/dreamer" || exit 1

# 日志目录
LOGDIR="./logs_metadrive_on_ramp/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"

# 运行训练（相当于之前的 python dreamerv3/main.py）
python3 dreamerv3/main.py \
  --configs metadrive_on_ramp \
  --logdir "$LOGDIR" \
  --run.log_every 120 \
  --run.report_every 300 \
  --run.save_every 900 \
  --run.steps 1e6 \
  --batch_size 4 \
  --jax.platform cuda

