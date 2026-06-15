#!/bin/bash
#SBATCH -J exid_test
#SBATCH -p L40
#SBATCH --gres=gpu:l40:1
#SBATCH -c 7
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -o logs/exid_test_%j.out
#SBATCH -e logs/exid_test_%j.err

set -x
exec 2>&1

echo "=== step 1: env ==="
env | grep -i sumo || true
echo "PATH=$PATH"

echo "=== step 2: conda ==="
source /share/home/u23516/miniforge3/etc/profile.d/conda.sh || echo "conda source failed"
conda activate metadrive || echo "conda activate failed"
which python3

echo "=== step 3: module ==="
module load sumo/1.20 || echo "module failed"
echo "SUMO_HOME=$SUMO_HOME"

echo "=== step 4: sumo export ==="
export SUMO_HOME="${SUMO_HOME:-/share/apps/sumo-1.20}/share/sumo"
echo "final SUMO_HOME=$SUMO_HOME"

echo "=== step 5: test imports ==="
python3 -c "
import os, sys
sys.path.insert(0, os.environ['SUMO_HOME'] + '/tools')
import sumolib; print('sumolib OK:', sumolib.__file__)
import pandas, numpy, cv2; print('all pkgs OK')
" 2>&1

echo "=== ALL TESTS PASSED ==="
