#!/bin/bash
#SBATCH --job-name=sel_traj
#SBATCH --comment="Select balanced trajectories for online Dreamer training"
#SBATCH --partition=intel
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/sel_traj_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/sel_traj_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

echo "=== Select Online Training Trajectories ==="
echo "Start: $(date)"

python -u select_online_trajectories.py \
    --merge-cache /share/home/u23516/code/meta_dreamer-main/mirro_data_map/exid_merge_cache.json \
    --data-dir /share/home/u23516/data/exiD-dataset-v2.1/data \
    --train-locs 0 2 4 5 6 \
    --max-per-loc 15 \
    --output /share/home/u23516/code/meta_dreamer-main/mirro_data_map/exid_online_selection.json \
    --plot-dir /share/home/u23516/code/meta_dreamer-main/pytorch/logs/plots_online_selection \
    --cache-stats /share/home/u23516/code/meta_dreamer-main/pytorch/logs/online_traj_stats.json

echo "Done: $(date)"
