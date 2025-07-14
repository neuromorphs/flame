#!/usr/bin/bash

set -x

additional_args=""
if [ $# -ne 0 ]; then
    additional_args="$1"
fi

echo "additional_args: $additional_args"

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

B_TOKENS=15
LR=3e-4
SEQLEN=2k
STEPS=$((1271*$B_TOKENS))
WANDB_NAME=pure-340M-${B_TOKENS}b-lr${LR}-seq${SEQLEN}
DUMPFOLDER=exp/hgrn-pure-340M-seq${SEQLEN}/${B_TOKENS}b.steps${STEPS}.lr${LR}

# set bsz, seqlen, context based on seqlen
if [ $SEQLEN == "2k" ]; then
    BATCHSIZE=48
    SEQLEN_INT=2048
    CONTEXTLEN=2048
elif [ $SEQLEN == "4k" ]; then
    BATCHSIZE=24
    SEQLEN_INT=4096
    CONTEXTLEN=4096
else
    echo "Invalid SEQLEN: $SEQLEN"
    exit 1
fi

echo "STEPS: $STEPS"
echo "LR: $LR"
echo "B_TOKENS: $B_TOKENS"
echo "SEQLEN: $SEQLEN"
echo "WANDB_NAME: $WANDB_NAME"
echo "DUMPFOLDER: $DUMPFOLDER"
echo "BATCHSIZE: $BATCHSIZE"
echo "CONTEXTLEN: $CONTEXTLEN"
echo "SEQLEN_INT: $SEQLEN_INT"

bash train.sh \
  --job.config_file train_configs/hgrn.toml \
  --model.config configs/hgrn_340M.json \
  --model.tokenizer_path mistralai/Mistral-7B-v0.1 \
  --training.steps $STEPS \
  --training.dataset HuggingFaceTB/smollm-corpus \
  --training.dataset_name fineweb-edu-dedup \
  --training.dataset_split train \
  --job.dump_folder $DUMPFOLDER \
  --metrics.save_tb_folder $DUMPFOLDER \
  --checkpoint.load_step -1 \
  --checkpoint.keep_latest_k 10 \
  --checkpoint.interval 2048 \
  --training.batch_size $BATCHSIZE \
  --training.seq_len $SEQLEN_INT \
  --training.context_len $CONTEXTLEN \
  $additional_args
