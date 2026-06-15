#!/bin/bash

#SBATCH --job-name=md_render
#SBATCH --comment="MetaDrive 3D rendering via Xvnc virtual display"
#SBATCH --partition=L40
#SBATCH --time=0-0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40:1
#SBATCH --mem=32g
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/logs/%x_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/logs/%x_%j.err

cd /share/home/u23516/code/meta_dreamer-main || exit 1
mkdir -p logs/metadrive_engine

echo "=== Environment ==="
echo "HOSTNAME: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no gpu"

# Activate metadrive env
source /share/home/u23516/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
echo "Python: $(which python)"

# Start virtual display via Xvnc
DISPLAY_NUM=99
echo "Starting Xvnc on :${DISPLAY_NUM}..."
Xvnc :${DISPLAY_NUM} -geometry 1920x1080 -depth 24 -SecurityTypes None &
XVNC_PID=$!
sleep 2
export DISPLAY=:${DISPLAY_NUM}

# Verify display
xdpyinfo 2>&1 | head -5 || echo "xdpyinfo failed"

export PYTHONUNBUFFERED=1

echo "=== Starting render ==="
python dreamer/tools/render_metadrive_engine.py \
    --scenario logs/map_features/navsim_scenario_test.pkl \
    --output logs/metadrive_engine/ \
    --max_steps 100 \
    --fps 12

EXIT_CODE=$?

# Cleanup
kill $XVNC_PID 2>/dev/null

echo "=== Done (exit: ${EXIT_CODE}) ==="
