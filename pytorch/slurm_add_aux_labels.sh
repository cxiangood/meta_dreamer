#!/bin/bash
#SBATCH -J aux_labels
#SBATCH -p CPU
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH -o logs/aux_labels_loc%a.out
#SBATCH -e logs/aux_labels_loc%a.err
#SBATCH --array=0-6

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Run with: sbatch --array=0-6 slurm_add_aux_labels.sh"
    exit 1
fi

LOC_ID=$SLURM_ARRAY_TASK_ID

source /share/home/u23516/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
module load sumo/1.20
export SUMO_HOME="${SUMO_HOME}/share/sumo"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"

cd /share/home/u23516/code/meta_dreamer-main/pytorch
mkdir -p logs

echo "=== Generating aux labels for loc=${LOC_ID} ==="
echo "Start: $(date)"

python aux_data/add_aux_labels.py \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --map-dir /share/home/u23516/code/meta_dreamer-main/mirro_data_map \
    --loc "$LOC_ID" 2>&1

echo "Done: $(date)"
