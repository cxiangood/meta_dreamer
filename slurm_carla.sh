#!/bin/bash
#SBATCH --job-name=carla-dreamerv3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=carla_training_%j.out
#SBATCH --error=carla_training_%j.err

# =====================================
# SLURM作业脚本 - CARLA + DreamerV3训练
# =====================================

echo "作业开始时间: $(date)"
echo "节点: $SLURM_JOB_NODELIST"
echo "作业ID: $SLURM_JOB_ID"

# 1. 加载模块
module load cuda/11.8
module load python/3.9
module load gcc/9.3.0

# 2. 激活Python环境
source ~/miniconda3/bin/activate carla_env

# 3. 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

# 4. 配置虚拟显示 (关键!)
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!
sleep 5

# 5. CARLA无头模式配置
export UE4_ROOT=""
export CARLA_SERVER=""
export SDL_VIDEODRIVER=offscreen

# 6. 启动CARLA服务器 (后台运行)
echo "启动CARLA服务器..."
cd $HOME/CARLA_0.9.15
./CarlaUE4.sh -opengl -quality-level=Low -world-port=2000 -rpc-port=2000 &
CARLA_PID=$!

# 等待CARLA启动
sleep 30

# 7. 启动Python服务端
echo "启动Python服务端..."
cd $SLURM_SUBMIT_DIR/python37
python carla0915.py &
SERVER_PID=$!

# 等待服务端启动
sleep 10

# 8. 开始训练
echo "开始DreamerV3训练..."
cd $SLURM_SUBMIT_DIR/dreamerv3/dreamerv3
python main.py --configs carla0915 --task carla0915_keeping \
    --logdir ~/logdir/carla_supercomputer_${SLURM_JOB_ID} \
    --run.steps 1000000

# 9. 清理进程
echo "清理进程..."
kill $SERVER_PID 2>/dev/null
kill $CARLA_PID 2>/dev/null
kill $XVFB_PID 2>/dev/null

echo "作业结束时间: $(date)"