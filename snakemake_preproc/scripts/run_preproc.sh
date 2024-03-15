#!/bin/bash
#SBATCH --account=def-akhanf-ab
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=2:00
echo "begin"
source $(kpy _kpy_wrapper)
kpy load venv_train_degad
echo "beginning training"
cd /project/6050199/akhanf/cfmm-bids/data/Lau/degad/snakemake/preproc
snakemake --cores 1
