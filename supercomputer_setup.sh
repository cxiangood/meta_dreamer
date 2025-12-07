#!/bin/bash
# 超算上运行CARLA + DreamerV3的部署脚本

# =====================================
# 超算CARLA强化学习部署指南
# =====================================

echo "=== 超算CARLA强化学习环境配置 ==="

# 1. 检查超算环境
check_environment() {
    echo "检查超算环境..."
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ 检测到GPU:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    else
        echo "✗ 未检测到GPU"
    fi
    
    # 检查显示支持
    if [ -n "$DISPLAY" ]; then
        echo "✓ 检测到显示环境: $DISPLAY"
    else
        echo "⚠ 未检测到显示环境，需要配置虚拟显示"
    fi
    
    # 检查Docker支持
    if command -v docker &> /dev/null; then
        echo "✓ 检测到Docker"
    else
        echo "⚠ 未检测到Docker，建议使用Singularity"
    fi
}

# 2. 配置虚拟显示 (关键!)
setup_virtual_display() {
    echo "配置虚拟显示..."
    
    # 方法1: Xvfb (推荐用于超算)
    if command -v Xvfb &> /dev/null; then
        echo "使用Xvfb配置虚拟显示"
        export DISPLAY=:99
        Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
        sleep 2
    fi
    
    # 方法2: 使用EGL (无头渲染)
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    export PYOPENGL_PLATFORM=egl
}

# 3. 环境模块加载 (根据超算系统调整)
load_modules() {
    echo "加载必要模块..."
    
    # 常见超算模块系统
    if command -v module &> /dev/null; then
        module load cuda/11.8
        module load python/3.9
        module load gcc/9.3.0
        module load cmake/3.20
        # 根据你的超算系统调整
    fi
    
    # 或者使用conda环境
    if command -v conda &> /dev/null; then
        conda activate carla_env  # 你的环境名
    fi
}

check_environment
setup_virtual_display
load_modules