#!/bin/bash

for chunk in 0 1 2 3 4
do
    python src/compute_baselines.py \
                --path_to_tokenized_data="./data/final_chunks/gutenberg_${chunk}_min_tokens5000_seed42" \
                --path_to_target_model='./pretrained/models/open_llama_7b' \
                --path_to_target_tokenizer='./pretrained/tokenizers/open_llama_7b' \
                --path_to_reference_model='openai-community/gpt2' \
                --path_to_reference_tokenizer='openai-community/gpt2' \
                --output_dir="./baseline_results" --suffix="gpt2_128_gutenberg_${chunk}" \
                --seq_length=128 --stride=127 --seed=42
done
