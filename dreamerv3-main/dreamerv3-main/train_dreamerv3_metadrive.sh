#!/bin/bash

# DreamerV3 MetaDrive Lane Keeping 训练启动脚本
echo "=== DreamerV3 MetaDrive Lane Keeping Training ==="

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" != *"dreamerv3"* ]]; then
    echo "警告：未检测到dreamerv3虚拟环境，尝试激活..."
    source ~/smart_driving/Prj_worldmoudle/dreamerv3/bin/activate
fi

# 进入项目目录
cd ~/smart_driving/Prj_worldmoudle/dreamerv3-main/dreamerv3-main

echo "当前环境：$VIRTUAL_ENV"
echo "当前目录：$(pwd)"

# 设置环境变量优化GPU内存
export CUDA_VISIBLE_DEVICES=0  # 如果有GPU
export TF_CPP_MIN_LOG_LEVEL=2  # 减少TensorFlow日志
export JAX_ENABLE_X64=false  # 使用float32节省内存

# JAX / XLA 内存相关设置，帮助避免显存预分配和 OOM
# 说明：可以把 XLA_PYTHON_CLIENT_MEM_FRACTION 调小到 0.4-0.7 之间根据显存调整
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}

# 如果设置了 DEBUG_CPU=1，则强制在 CPU 上运行（便于调试、避免 GPU OOM）
if [[ "${DEBUG_CPU}" == "1" ]]; then
    echo "DEBUG_CPU=1: 强制使用 CPU（JAX_PLATFORM_NAME=cpu），并清除 CUDA 可见设备。"
    export JAX_PLATFORM_NAME=cpu
    export CUDA_VISIBLE_DEVICES=""
fi

# 可调参数：批大小、批长度、并行副本数（可以在运行前覆盖这些环境变量）
BATCH_SIZE=${BATCH_SIZE:-2}
BATCH_LENGTH=${BATCH_LENGTH:-16}
REPLICAS=${REPLICAS:-1}

echo "设置的环境变量："
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  JAX_ENABLE_X64=$JAX_ENABLE_X64"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE"
echo "  XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "训练超参（可通过环境变量覆盖）："
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  BATCH_LENGTH=$BATCH_LENGTH"
echo "  REPLICAS=$REPLICAS"
echo ""
# 如果系统上有 nvidia-smi，可检查可用显存并自动调整内存 fraction / batch
if command -v nvidia-smi &> /dev/null && [[ -z "$JAX_PLATFORM_NAME" || "$JAX_PLATFORM_NAME" != "cpu" ]]; then
    # 取得第一个 GPU 的空闲显存（MB）
    FREE_MEM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '\r') || true
    if [[ "$FREE_MEM_MB" =~ ^[0-9]+$ ]]; then
        echo "检测到 GPU 可用显存: ${FREE_MEM_MB} MB，正在根据显存调整 XLA 内存 fraction / batch 建议..."
        # 如果小于 6GB，使用更小的 fraction 和更保守的批量
        if [ "$FREE_MEM_MB" -lt 6000 ]; then
            export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.25}
            if [ "$BATCH_SIZE" -gt 1 ]; then
                echo "将 BATCH_SIZE 降低为 1 以降低显存压力"
                BATCH_SIZE=1
            fi
        elif [ "$FREE_MEM_MB" -lt 10000 ]; then
            export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.3}
        else
            export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}
        fi
        echo "  使用 XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION, BATCH_SIZE=$BATCH_SIZE"
    else
        echo "无法解析 nvidia-smi 输出 (FREE_MEM_MB='$FREE_MEM_MB')，保持 XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION"
    fi
fi

echo "开始DreamerV3训练..."
echo "配置：metadrive_lane_keeping"
echo "命令：python3 dreamerv3/main.py --configs metadrive_lane_keeping --logdir ./logs"

# 启动训练，静默模式运行，重要信息保存到日志
echo "启动DreamerV3训练（静默模式，日志保存到training.log）"
echo "可以用 'tail -f training.log' 查看训练进度"

# 创建一个过滤脚本
cat > filter_output.py << 'EOF'
import sys
import re

# 定义需要保留的重要信息模式
important_patterns = [
    r'loss', r'reward', r'episode', r'step', r'fps', 
    r'error', r'Error', r'warn', r'Warn', r'INFO',
    r'DREAMER-ACTION',
    r'completed', r'Done', r'finished', r'training',
    r'Replica:', r'Logdir:', r'Run script:',
    r'===', r'Compiling', r'Largest checkpoint'
]

# 定义需要过滤掉的模式
filter_patterns = [
    r'^\[.*\]$', r'^[0-9]+$', r'\.count$', 
    r'kernel', r'bias', r'scale', r'linear', 
    r'conv', r'norm', r'mlp', r'head', r'pred', 
    r'^/.*/', r'^\s*$'
]

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    
    # 检查是否需要过滤
    should_filter = any(re.search(pattern, line, re.IGNORECASE) for pattern in filter_patterns)
    
    if not should_filter:
        # 检查是否是重要信息
        is_important = any(re.search(pattern, line, re.IGNORECASE) for pattern in important_patterns)
        if is_important or len(line) > 50:  # 保留长行，通常是重要信息
            print(line)
            sys.stdout.flush()
EOF

# 启动训练并应用过滤，设置 JAX/XLA 环境以减少内存预分配并传入较小的 batch/replica
echo "启动命令中注入 XLA/JAX 环境并传入 batch/replica 参数。"

# If user requested combined rendering + print, enable DREAMER_PRINT_ACTIONS and run in debug (non-parallel) mode
if [[ "${RENDER_AND_PRINT}" == "1" ]]; then
    echo "RENDER_AND_PRINT=1: 将以 debug(非并行) 模式运行并启用 DREAMER_ACTION 打印。"
    export DREAMER_PRINT_ACTIONS=1
    # Pass debug into the 'run' section of the config as a boolean so elements.Flags accepts it.
    PY_DEBUG_ARG="--run.debug True"
else
    PY_DEBUG_ARG=""
fi

JAX_PLATFORMS="" \
XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE \
XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_PYTHON_CLIENT_MEM_FRACTION \
JAX_ENABLE_X64=$JAX_ENABLE_X64 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python3 dreamerv3/main.py --configs metadrive_lane_keeping --logdir ./logs \
    --batch_size $BATCH_SIZE --batch_length $BATCH_LENGTH --replicas $REPLICAS $PY_DEBUG_ARG 2>&1 | \
python filter_output.py | tee training.log