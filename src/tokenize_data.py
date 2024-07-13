from transformers import LlamaTokenizer
from datasets import load_from_disk
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--path_to_tokenizer", type=str, required=True)
parser.add_argument("--path_to_dataset", required=False, type=str)
parser.add_argument("--nb_workers", type=int, default=10)
parser.add_argument("--max_shard_size", type=str, default="4GB")
args = parser.parse_args()

def main():
    DATA_DIR = args.data_dir
    PATH_TO_DATASET = args.path_to_dataset
    DATASET_NAME = PATH_TO_DATASET.split('/')[-1]
    PATH_TO_TOKENIZER = args.path_to_tokenizer
    TOKENIZER_NAME = PATH_TO_TOKENIZER.split('/')[-1]
    
    tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_TOKENIZER)
    
    print(f"Loading {DATASET_NAME}...")
    dataset = load_from_disk(PATH_TO_DATASET)
    
    print(f"Starting tokenization {DATASET_NAME}...")
    tokenized_dataset = dataset.map(
                lambda samples: tokenizer(samples["text"]),
                batched=False,
                num_proc=args.nb_workers,
                remove_columns=["text"],
                load_from_cache_file=False,
                desc=f"Running {TOKENIZER_NAME} tokenizer on {DATASET_NAME}",
            )
    
    print(f"Tokenization done for {DATASET_NAME}...")

    tokenized_output_dir = f"{DATA_DIR}/tokenized/{TOKENIZER_NAME}"
    tokenized_dataset.save_to_disk(
            f"{tokenized_output_dir}/{DATASET_NAME}", max_shard_size=args.max_shard_size, num_proc=args.nb_workers
            )
    
    print(f"Tokenized dataset saved for {DATASET_NAME}")

if __name__ == "__main__":
    main()
