#!/bin/bash
#SBATCH --job-name=hgrn_340M
#SBATCH --partition=minor-use-case
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -x
bash train-hgrn.sh