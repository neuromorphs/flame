#!/bin/bash
#SBATCH --job-name=hgrn_340M
#SBATCH --partition=minor-use-case
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -x

# Activate virtual environment
source .venv/bin/activate

# Load environment variables
source .env

# Print GPU status
nvidia-smi

#####################################
# to decide the steps:
#####################################
# SEQLEN=2k
### 2048*48*8=786,432 tokens per step
### 1,271 steps = 1B tokens
### 19,074 steps = 15B tokens
#####################################

SEQLEN=2k
STEPS=2048
LR=3e-4

DUMPFOLDER=exp/hgrn-340M-${SEQLEN}-pure/steps${STEPS}.lr${LR}
bash train.sh \
  --job.config_file train_configs/hgrn_${SEQLEN}.toml \
  --model.config fla-hub/transformer-1.3B-100B \
  --model.tokenizer_path mistralai/Mistral-7B-v0.1 \
  --training.steps $STEPS \
  --training.dataset HuggingFaceTB/smollm-corpus \
  --training.dataset_name fineweb-edu-dedup \
  --training.dataset_split train \
  --job.dump_folder $DUMPFOLDER \
  --metrics.save_tb_folder $DUMPFOLDER \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 10 \
  --checkpoint.interval 2048
