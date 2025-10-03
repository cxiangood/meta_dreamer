#!/bin/bash

# 定义项目和环境路径
PROJECT_DIR="$HOME/smart_driving/Prj_worldmoudle"
VENV_DIR="$PROJECT_DIR/dreamerv3"
CODE_DIR="$PROJECT_DIR/dreamerv3-main/dreamerv3-main"

# 检查虚拟环境是否存在
if [ ! -d "$VENV_DIR" ]; then
    echo "错误：虚拟环境 $VENV_DIR 不存在！"
    exit 1
fi

# 检查激活脚本是否存在
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "错误：未找到激活脚本 $VENV_DIR/bin/activate"
    exit 1
fi

# 进入项目目录并激活环境
echo "进入项目目录..."
cd "$PROJECT_DIR" || { echo "无法进入项目目录 $PROJECT_DIR"; exit 1; }

echo "激活虚拟环境..."
source "$VENV_DIR/bin/activate" || { echo "激活环境失败"; exit 1; }

# 进入代码目录
if [ -d "$CODE_DIR" ]; then
    echo "进入代码目录..."
    cd "$CODE_DIR" || { echo "无法进入代码目录 $CODE_DIR"; exit 1; }
else
    echo "警告：代码目录 $CODE_DIR 不存在，将留在项目根目录"
fi

echo "环境准备就绪！当前路径：$(pwd)"
