#!/bin/bash

# CARLA 低配启动脚本 - 解决GPU驱动问题
# 使用方法: ./start_carla_safe.sh

echo "=== CARLA 安全启动脚本 ==="
echo "适用于GPU驱动有问题的情况"
echo

# 检测GPU状态
check_gpu() {
    if nvidia-smi > /dev/null 2>&1; then
        echo "✓ NVIDIA GPU可用"
        return 0
    else
        echo "⚠ NVIDIA GPU不可用，使用CPU渲染"
        return 1
    fi
}

# 设置环境变量
setup_environment() {
    echo "设置环境变量..."
    
    # 强制软件渲染
    export LIBGL_ALWAYS_SOFTWARE=1
    export MESA_GL_VERSION_OVERRIDE=4.5
    export MESA_GLSL_VERSION_OVERRIDE=450
    
    # 禁用硬件加速
    export GALLIUM_DRIVER=llvmpipe
    export LP_NUM_THREADS=4
    
    # UE4 设置
    export UE4_ROOT=""
    export MALLOC_CHECK_=0
    
    echo "✓ 环境变量设置完成"
}

# 清理之前的CARLA进程
cleanup_carla() {
    echo "清理之前的CARLA进程..."
    pkill -f CarlaUE4 2>/dev/null || true
    sleep 2
    echo "✓ 清理完成"
}

# 启动CARLA (方案1: 完全无渲染)
start_carla_nullrhi() {
    echo "启动CARLA (无渲染模式)..."
    ./CarlaUE4.sh -nullrhi -no-rendering-mode -world-port=2000 -rpc-port=2000 &
    CARLA_PID=$!
    echo "CARLA PID: $CARLA_PID"
    
    # 等待启动
    sleep 10
    
    # 检查是否成功启动
    if kill -0 $CARLA_PID 2>/dev/null; then
        echo "✓ CARLA启动成功 (PID: $CARLA_PID)"
        return 0
    else
        echo "✗ CARLA启动失败"
        return 1
    fi
}

# 启动CARLA (方案2: OpenGL软件渲染)
start_carla_opengl() {
    echo "启动CARLA (OpenGL软件渲染)..."
    ./CarlaUE4.sh -opengl -quality-level=Low -no-rendering-mode -benchmark -fps=5 -world-port=2000 -rpc-port=2000 &
    CARLA_PID=$!
    echo "CARLA PID: $CARLA_PID"
    
    # 等待启动
    sleep 10
    
    # 检查是否成功启动
    if kill -0 $CARLA_PID 2>/dev/null; then
        echo "✓ CARLA启动成功 (PID: $CARLA_PID)"
        return 0
    else
        echo "✗ CARLA启动失败"
        return 1
    fi
}

# 验证CARLA是否可连接
test_carla_connection() {
    echo "测试CARLA连接..."
    
    # 使用Python测试连接
    python3 << 'EOF'
import sys
try:
    import carla
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    version = client.get_server_version()
    print(f"✓ CARLA连接成功，版本: {version}")
    sys.exit(0)
except Exception as e:
    print(f"✗ CARLA连接失败: {e}")
    sys.exit(1)
EOF
}

# 主程序
main() {
    echo "开始启动CARLA..."
    
    # 检查CARLA是否存在
    if [ ! -f "./CarlaUE4.sh" ]; then
        echo "✗ 错误: 未找到 CarlaUE4.sh"
        echo "请确保在CARLA目录中运行此脚本"
        exit 1
    fi
    
    setup_environment
    cleanup_carla
    
    # 尝试方案1: 完全无渲染
    echo "=== 尝试方案1: 完全无渲染模式 ==="
    if start_carla_nullrhi; then
        if test_carla_connection; then
            echo "🎉 CARLA启动成功！"
            echo "可以开始训练了"
            wait $CARLA_PID
            exit 0
        fi
    fi
    
    echo "方案1失败，清理进程..."
    cleanup_carla
    
    # 尝试方案2: OpenGL软件渲染
    echo "=== 尝试方案2: OpenGL软件渲染 ==="
    if start_carla_opengl; then
        if test_carla_connection; then
            echo "🎉 CARLA启动成功！"
            echo "可以开始训练了"
            wait $CARLA_PID
            exit 0
        fi
    fi
    
    echo "所有方案都失败了 😞"
    echo "请检查："
    echo "1. CARLA是否正确安装"
    echo "2. 系统是否有足够的内存"
    echo "3. 是否有其他CARLA进程在运行"
    
    cleanup_carla
    exit 1
}

# 信号处理
cleanup() {
    echo -e "\n收到中断信号，清理进程..."
    cleanup_carla
    exit 0
}

trap cleanup INT TERM

# 运行主程序
main "$@"