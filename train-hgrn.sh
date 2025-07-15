#!/usr/bin/bash

set -x

# Default values
MODELNAME=pure  # [pure, hybrid]
MODELSIZE=340M  # [170M, 340M]
LR=3e-4
SEQLEN=2k  # [2k, 4k]
B_TOKENS=15  # NOTE! this assumes the default hgrn.toml

# Parse command line arguments
while getopts "m:s:l:q:b:h" opt; do
    case $opt in
        m)
            MODELNAME="$OPTARG"
            ;;
        s)
            MODELSIZE="$OPTARG"
            ;;
        l)
            LR="$OPTARG"
            ;;
        q)
            SEQLEN="$OPTARG"
            ;;
        b)
            B_TOKENS="$OPTARG"
            ;;
        h)
            echo "Usage: $0 [OPTIONS] [-- additional_args]"
            echo "Options:"
            echo "  -m MODEL_NAME   Model name: pure or hybrid (default: pure)"
            echo "  -s MODEL_SIZE   Model size: 170M or 340M (default: 340M)"
            echo "  -l LEARNING_RATE Learning rate (default: 3e-4)"
            echo "  -q SEQ_LEN      Sequence length: 2k or 4k (default: 2k)"
            echo "  -b B_TOKENS     B tokens (default: 15)"
            echo "  -h              Show this help message"
            echo ""
            echo "Use -- to separate script options from additional arguments:"
            echo "Example: $0 -m hybrid -s 170M -- --some-additional-flag"
            echo "Example: $0 -m hybrid -s 170M -l 5e-4 -q 4k -b 10"
            exit 0
            ;;
        \?)
            # Check if this is the start of additional arguments
            if [[ "${!OPTIND}" == --* ]]; then
                # This looks like additional arguments, stop parsing options
                ((OPTIND--))
                break
            else
                echo "Invalid option: -$OPTARG" >&2
                echo "Use -h for help, or use -- to separate options from additional arguments"
                exit 1
            fi
            ;;
    esac
done

# Shift past the processed options
shift $((OPTIND-1))

# Remaining arguments are additional_args
additional_args="$*"

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

# Calculate base steps and round up to next multiple of 2048
STEPS_BASE=$((1271*$B_TOKENS))
STEPS=$((2048 * (($STEPS_BASE + 2047) / 2048)))
WANDB_NAME=${MODELNAME}-${MODELSIZE}-${B_TOKENS}b-lr${LR}-seq${SEQLEN}
DUMPFOLDER=exp/hgrn-${MODELNAME}-${MODELSIZE}-seq${SEQLEN}/${B_TOKENS}b.steps${STEPS}.lr${LR}

export WANDB_NAME=${WANDB_NAME}

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

# set model config (pure or hybrid)
if [ $MODELNAME == "pure" ]; then
    MODELCONFIG=configs/hgrn_${MODELSIZE}.json
else
    MODELCONFIG=configs/hgrn_hyb_${MODELSIZE}.json
fi

echo "MODELNAME: $MODELNAME"
echo "MODELSIZE: $MODELSIZE"
echo "MODELCONFIG: $MODELCONFIG"
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
  --model.config $MODELCONFIG \
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
