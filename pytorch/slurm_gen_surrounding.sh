#!/bin/bash
#SBATCH --job-name=gen_surround
#SBATCH --comment="Generate _surrounding.npz companion files for VehicleHead training"
#SBATCH --partition=amd
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/gen_surround_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-main/pytorch/logs/gen_surround_%j.err

mkdir -p /share/home/u23516/code/meta_dreamer-main/pytorch/logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate metadrive
cd /share/home/u23516/code/meta_dreamer-main/pytorch

echo "=== Generating _surrounding.npz files ==="
echo "Start: $(date)"

python -u aux_data/add_surrounding_vehicles.py \
    --npz-dir /share/home/u23516/data/exid_dreamer_data \
    --exid-dir /share/home/u23516/data/exiD-dataset-v2.1 \
    --n-vehicles 5

echo "Done: $(date)"
