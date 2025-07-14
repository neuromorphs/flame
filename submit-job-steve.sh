#!/bin/bash
#SBATCH --job-name=hgrn_340M_2k_pure
#SBATCH --partition=gpu_h100
#SBATCH --output=hgrn_%j.log
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -x
module load 2024
module load CUDA/12.6.0
NGPU=4 bash train-hgrn.sh
