# Load model directly
from transformers import LlamaTokenizer, LlamaForCausalLM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False, default="openlm-research/open_llama_7b")
parser.add_argument( "--write_dir", type=str, required=True)
parser.add_argument("--max_shard_size", type=str, default="4GB")

args = parser.parse_args()

MODEL_NAME = args.model
WRITE_DIR = args.write_dir
MAX_SHARD_SIZE = args.max_shard_size

def main():
    
    # load and write the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(f"{WRITE_DIR}/tokenizers/{MODEL_NAME.split('/')[1]}")
    print(f'The tokenizer for {MODEL_NAME} has been saved successfully.')
    
    # load and write the model
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME)
    model.save_pretrained(
        f"{WRITE_DIR}/models/{MODEL_NAME.split('/')[1]}", max_shard_size=MAX_SHARD_SIZE
    )
    print(f'The model {MODEL_NAME} has been saved successfully.')

if __name__ == "__main__":
    main()
