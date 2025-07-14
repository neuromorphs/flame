from datasets import load_dataset

# load fineweb-edu with parallel processing
# load only the first 100 files (out of 234)
dataset = load_dataset(
    "HuggingFaceTB/smollm-corpus", 
    name="fineweb-edu-dedup", 
    num_proc=64, 
    cache_dir="/scratch-shared/sabreu/cache",
    data_files="fineweb-edu-dedup/train-000*-of-00234.parquet"
)
