#!/bin/bash

# CARLA显存监控脚本

echo "🖥️  CARLA显存占用监控"
echo "===================="

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi未找到，检查集成显卡或CPU显存..."
    
    # 检查系统内存
    echo "📊 系统内存使用情况："
    free -h
    
    # 检查CARLA进程内存占用
    echo -e "\n🔍 CARLA进程内存占用："
    ps aux | grep -E "(CarlaUE4|carla)" | grep -v grep || echo "未找到CARLA进程"
    
    exit 0
fi

# GPU显存监控函数
monitor_gpu() {
    echo "🎮 GPU显存监控 (每3秒刷新一次，按Ctrl+C停止)"
    echo "================================================"
    
    while true; do
        clear
        echo "🖥️  CARLA显存占用监控 - $(date '+%H:%M:%S')"
        echo "================================================"
        
        # 显示GPU基本信息
        echo "📊 GPU状态概览："
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw \
                   --format=csv,noheader,nounits | while IFS=',' read -r idx name temp util mem_used mem_total power; do
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "0")
            
            echo "  GPU $idx: $name"
            echo "    温度: ${temp}°C | 利用率: ${util}% | 功耗: ${power}W"
            echo "    显存: ${mem_used}MB / ${mem_total}MB (${mem_percent}%)"
            echo
        done
        
        # 显示详细显存使用
        echo "🔍 详细显存分配："
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=',' read -r pid process_name used_mem; do
            if [[ -n "$pid" ]]; then
                pid=$(echo $pid | xargs)
                process_name=$(echo $process_name | xargs)
                used_mem=$(echo $used_mem | xargs)
                echo "  PID $pid: $process_name - ${used_mem}MB"
            fi
        done
        
        # 查找CARLA相关进程
        echo -e "\n🚗 CARLA进程监控："
        carla_pids=$(pgrep -f CarlaUE4 2>/dev/null)
        if [[ -n "$carla_pids" ]]; then
            for pid in $carla_pids; do
                if [[ -f "/proc/$pid/status" ]]; then
                    cmd=$(ps -p $pid -o command --no-headers 2>/dev/null | cut -c1-50)
                    mem_info=$(cat /proc/$pid/status 2>/dev/null | grep -E "VmRSS|VmSize")
                    echo "  PID $pid: $cmd"
                    echo "    $mem_info" | sed 's/^/      /'
                fi
            done
        else
            echo "  未找到CARLA进程"
        fi
        
        # 显示系统内存
        echo -e "\n💾 系统内存："
        free -h | head -2 | tail -1 | awk '{print "  已用: "$3" / 总计: "$2" ("int($3/$2*100)"%)"}' 
        
        echo -e "\n按 Ctrl+C 停止监控..."
        sleep 3
    done
}

# 单次显存检查函数
check_once() {
    echo "📊 当前显存使用状态："
    echo "===================="
    
    # GPU信息
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
    while IFS=',' read -r idx name mem_used mem_total; do
        mem_used=$(echo $mem_used | xargs)
        mem_total=$(echo $mem_total | xargs)
        mem_free=$((mem_total - mem_used))
        mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "0")
        
        echo "GPU $idx ($name):"
        echo "  已用显存: ${mem_used}MB"
        echo "  空闲显存: ${mem_free}MB" 
        echo "  总显存: ${mem_total}MB"
        echo "  使用率: ${mem_percent}%"
        echo
    done
    
    # CARLA进程检查
    echo "🔍 CARLA进程显存占用："
    carla_found=false
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r pid process_name used_mem; do
        if [[ "$process_name" == *"CarlaUE4"* ]] || [[ "$process_name" == *"carla"* ]]; then
            carla_found=true
            echo "  PID $pid: $process_name - ${used_mem}MB"
        fi
    done
    
    if [[ "$carla_found" == "false" ]]; then
        echo "  未检测到CARLA GPU进程"
    fi
}

# 显存优化建议
show_optimization_tips() {
    echo -e "\n💡 显存优化建议："
    echo "=================="
    echo "1. 使用最低配置启动CARLA:"
    echo "   ./CarlaUE4.sh -opengl -quality-level=Low -no-rendering-mode"
    echo
    echo "2. 限制帧率和分辨率:"
    echo "   添加参数: -fps=10 -ResX=640 -ResY=480"
    echo
    echo "3. 如果仍然显存不足，尝试:"
    echo "   - 关闭其他GPU程序"
    echo "   - 重启系统清理显存"
    echo "   - 使用CPU渲染: -nullrhi"
    echo
    echo "4. 监控显存使用:"
    echo "   watch -n 1 nvidia-smi"
}

# 主程序
case "$1" in
    "monitor"|"-m")
        monitor_gpu
        ;;
    "once"|"-o"|"")
        check_once
        show_optimization_tips
        ;;
    "help"|"-h")
        echo "用法: $0 [选项]"
        echo "选项:"
        echo "  monitor, -m    实时监控显存使用"
        echo "  once, -o       单次检查显存状态 (默认)"
        echo "  help, -h       显示帮助信息"
        ;;
    *)
        echo "未知选项: $1"
        echo "使用 '$0 help' 查看帮助"
        exit 1
        ;;
esac