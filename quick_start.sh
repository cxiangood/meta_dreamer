#!/bin/bash

# 超算CARLA快速启动脚本
# 使用方法: ./quick_start.sh [GPU数量] [训练步数]

set -e

# 默认参数
NUM_GPUS=${1:-1}
TRAINING_STEPS=${2:-1000000}
PROJECT_DIR=$(pwd)

echo "================================================"
echo "超算CARLA自动驾驶训练 - 快速启动"
echo "================================================"
echo "GPU数量: $NUM_GPUS"
echo "训练步数: $TRAINING_STEPS" 
echo "项目目录: $PROJECT_DIR"
echo "================================================"

# 检查必要文件
check_files() {
    echo "检查必要文件..."
    
    local required_files=(
        "distributed_carla.py"
        "supercomputer_config.json"
        "python37/carla0915.py"
        "dreamerv3/dreamerv3/main.py"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            echo "错误: 缺少必要文件 $file"
            exit 1
        fi
        echo "  ✓ $file"
    done
}

# 检查CARLA安装
check_carla() {
    echo "检查CARLA安装..."
    
    local carla_paths=(
        "$HOME/CARLA_0.9.15/CarlaUE4.sh"
        "/opt/carla/CarlaUE4.sh"
        "./CarlaUE4.sh"
    )
    
    for path in "${carla_paths[@]}"; do
        if [[ -f "$path" ]]; then
            echo "  ✓ 找到CARLA: $path"
            CARLA_PATH="$path"
            return 0
        fi
    done
    
    echo "  ✗ 未找到CARLA安装"
    echo "请确保CARLA 0.9.15已正确安装"
    exit 1
}

# 检查Python环境
check_python() {
    echo "检查Python环境..."
    
    # 检查Python 3.7 (CARLA服务端)
    if command -v python3.7 &> /dev/null; then
        echo "  ✓ Python 3.7: $(python3.7 --version)"
    else
        echo "  ✗ 未找到Python 3.7"
        echo "CARLA服务端需要Python 3.7"
        exit 1
    fi
    
    # 检查Python 3.11 (DreamerV3)
    if command -v python3.11 &> /dev/null; then
        echo "  ✓ Python 3.11: $(python3.11 --version)"
    else
        echo "  ✗ 未找到Python 3.11" 
        echo "DreamerV3需要Python 3.11"
        exit 1
    fi
}

# 检查GPU
check_gpu() {
    echo "检查GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi -L | wc -l)
        echo "  ✓ 检测到 $gpu_count 个GPU"
        
        if [[ $gpu_count -lt $NUM_GPUS ]]; then
            echo "  ⚠ 请求的GPU数量($NUM_GPUS)超过可用数量($gpu_count)"
            NUM_GPUS=$gpu_count
            echo "  自动调整为 $NUM_GPUS 个GPU"
        fi
    else
        echo "  ✗ 未检测到NVIDIA GPU驱动"
        exit 1
    fi
}

# 设置虚拟显示
setup_display() {
    echo "设置虚拟显示..."
    
    # 检查是否已有Xvfb运行
    if pgrep -f "Xvfb :99" > /dev/null; then
        echo "  ✓ 虚拟显示已运行"
    else
        echo "  启动虚拟显示..."
        Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        sleep 3
        
        if pgrep -f "Xvfb :99" > /dev/null; then
            echo "  ✓ 虚拟显示启动成功"
        else
            echo "  ✗ 虚拟显示启动失败"
            exit 1
        fi
    fi
    
    export DISPLAY=:99
}

# 更新配置文件
update_config() {
    echo "更新配置文件..."
    
    # 备份原配置
    cp supercomputer_config.json supercomputer_config.json.bak
    
    # 使用Python更新配置
    python3 << EOF
import json

with open('supercomputer_config.json', 'r') as f:
    config = json.load(f)

# 更新配置
config['carla_path'] = '$CARLA_PATH'
config['num_carla_instances'] = $NUM_GPUS
config['training_steps'] = $TRAINING_STEPS
config['dreamerv3_settings']['run']['envs'] = $NUM_GPUS
config['dreamerv3_settings']['run']['steps'] = $TRAINING_STEPS

with open('supercomputer_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"  ✓ 配置已更新: {$NUM_GPUS}个环境, {$TRAINING_STEPS}训练步数")
EOF
}

# 创建日志目录
create_logdir() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local logdir="./logs/run_${timestamp}"
    
    mkdir -p "$logdir"
    echo "  ✓ 日志目录: $logdir"
    
    # 更新配置中的日志目录
    python3 << EOF
import json

with open('supercomputer_config.json', 'r') as f:
    config = json.load(f)

config['logdir'] = '$logdir'

with open('supercomputer_config.json', 'w') as f:
    json.dump(config, f, indent=2)
EOF
}

# 启动训练
start_training() {
    echo "启动分布式训练..."
    echo "================================================"
    
    # 设置环境变量
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
    export SDL_VIDEODRIVER=offscreen
    export __GL_SYNC_TO_VBLANK=0
    
    # 启动分布式训练脚本
    python3 distributed_carla.py \
        --config supercomputer_config.json \
        --num-instances $NUM_GPUS \
        --training-steps $TRAINING_STEPS
}

# 清理函数
cleanup() {
    echo -e "\n清理进程..."
    
    # 恢复配置文件
    if [[ -f supercomputer_config.json.bak ]]; then
        mv supercomputer_config.json.bak supercomputer_config.json
        echo "  ✓ 配置文件已恢复"
    fi
    
    # 杀死可能残留的进程
    pkill -f "CarlaUE4" || true
    pkill -f "carla0915.py" || true
    
    echo "  ✓ 清理完成"
}

# 设置信号处理
trap cleanup EXIT INT TERM

# 主流程
main() {
    check_files
    check_carla
    check_python
    check_gpu
    setup_display
    update_config
    create_logdir
    start_training
}

# 运行主程序
main "$@"