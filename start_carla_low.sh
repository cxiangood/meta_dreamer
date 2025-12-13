#!/bin/bash

# CARLA 最低配启动脚本

echo "🚀 启动CARLA最低配模式..."

# 设置环境变量
export UE4_ROOT=""
export MALLOC_CHECK_=0

# 检查CARLA路径
CARLA_PATHS=(
    "$HOME/CARLA_0.9.15/CarlaUE4.sh"
    "/opt/carla/CarlaUE4.sh"
    "./CarlaUE4.sh"
)

CARLA_EXE=""
for path in "${CARLA_PATHS[@]}"; do
    if [[ -f "$path" ]]; then
        CARLA_EXE="$path"
        echo "✓ 找到CARLA: $path"
        break
    fi
done

if [[ -z "$CARLA_EXE" ]]; then
    echo "❌ 未找到CARLA可执行文件"
    echo "请确保CARLA已安装在以下位置之一："
    for path in "${CARLA_PATHS[@]}"; do
        echo "  - $path"
    done
    exit 1
fi

# 清理之前的进程
echo "清理之前的CARLA进程..."
pkill -f CarlaUE4 2>/dev/null || true
sleep 2

# 启动CARLA
echo "启动CARLA (最低配置)..."
echo "命令: $CARLA_EXE -nullrhi -no-rendering-mode -benchmark -fps=5"

# 设置软件渲染环境变量 (强制CPU渲染)
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_GL_VERSION_OVERRIDE=4.5
export GALLIUM_DRIVER=llvmpipe
export LP_NUM_THREADS=4

"$CARLA_EXE" \
    -nullrhi \
    -no-rendering-mode \
    -benchmark \
    -fps=5 \
    -world-port=2000 \
    -rpc-port=2000 &

CARLA_PID=$!
echo "CARLA启动中... (PID: $CARLA_PID)"

# 等待启动
echo "等待CARLA服务器启动..."
for i in {1..30}; do
    if netstat -tlnp 2>/dev/null | grep -q ":2000 "; then
        echo "✅ CARLA服务器启动成功！"
        echo "端口2000已监听"
        break
    fi
    echo "等待中... ($i/30)"
    sleep 2
done

# 检查进程状态
if kill -0 $CARLA_PID 2>/dev/null; then
    echo "🎉 CARLA正在运行 (PID: $CARLA_PID)"
    echo "现在可以启动训练脚本了"
else
    echo "❌ CARLA启动失败"
    exit 1
fi

# 等待用户中断
echo "按 Ctrl+C 停止CARLA服务器..."
wait $CARLA_PID