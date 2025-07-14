#!/bin/bash
#SBATCH --job-name=hgrn_340M
#SBATCH --partition=minor-use-case
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
set -x

# Usage: bash train-hgrn.sh [OPTIONS] [-- additional_args]
# Options:
#   -m MODEL_NAME   Model name: pure or hybrid (default: pure)
#   -s MODEL_SIZE   Model size: 170M or 340M (default: 340M)
#   -l LEARNING_RATE Learning rate (default: 3e-4)
#   -q SEQ_LEN      Sequence length: 2k or 4k (default: 2k)
#   -b B_TOKENS     B tokens (default: 15)
# Use -- to separate script options from additional arguments:
# Example: bash train-hgrn.sh -m hybrid -s 170M -- --some-additional-flag
# Example: bash train-hgrn.sh -m hybrid -s 170M -l 5e-4 -q 4k -b 10

bash train-hgrn.sh -m pure -s 340M -l 3e-4 -q 2k -b 15