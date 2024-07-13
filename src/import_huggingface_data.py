from datasets import load_dataset, concatenate_datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hf_dataset", type=str, required=True) #(""pg19")
parser.add_argument("--write_dir", type=str, required=True)
parser.add_argument("--max_shard_size", type=str, default="4GB")
parser.add_argument("--hf_cache_dir", type=str, default=None)
args = parser.parse_args()

HF_DATASET = args.hf_dataset
WRITE_DIR = args.write_dir
MAX_SHARD_SIZE = args.max_shard_size
HF_CACHE_DIR = args.hf_cache_dir

def main():

    dataset = load_dataset(HF_DATASET, cache_dir = HF_CACHE_DIR)
    
    dataset = concatenate_datasets([dataset[key] for key in dataset.keys()])

    dataset.save_to_disk(
        f"{WRITE_DIR}/{HF_DATASET}",
        max_shard_size=MAX_SHARD_SIZE,
    )

if __name__ == "__main__":
    main()
