#!/bin/bash
#SBATCH --job-name=navsim_cfm_latent
#SBATCH --comment="NAVSIM offline CFM latent pretraining"
#SBATCH --partition=A800
#SBATCH --time=0-6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gres=gpu:a800:1
#SBATCH --mem=64g
#SBATCH --output=/share/home/u23516/code/meta_dreamer-sub/logs/%x_%j.out
#SBATCH --error=/share/home/u23516/code/meta_dreamer-sub/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1504047409@qq.com

set -euo pipefail

mkdir -p /share/home/u23516/code/meta_dreamer-sub/logs

if [ -f /share/home/u23516/miniforge3/etc/profile.d/conda.sh ]; then
  source /share/home/u23516/miniforge3/etc/profile.d/conda.sh
  conda activate metadrive || true
fi

export PYTHONPATH="/share/home/u23516/code/meta_dreamer-sub/dreamer:${PYTHONPATH:-}"
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

cd /share/home/u23516/code/meta_dreamer-sub

RUN_NAME=${RUN_NAME:-navsim_mini_cfm_latent_$(date +"%Y%m%d_%H%M%S")}
export RUN_NAME

bash /share/home/u23516/code/meta_dreamer-sub/run_navsim_cfm_latent_pretrain.sh
