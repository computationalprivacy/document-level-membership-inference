#!/bin/bash

for chunk in 0 1 2 3 4
do
    python src/compute_neighborhood_baseline.py \
                --path_to_balanced_data="./baseline_results/baseline_results_gpt2_128_gutenberg_${chunk}_balanced.csv" \
                --path_to_target_model='./pretrained/models/open_llama_7b' \
                --path_to_target_tokenizer='./pretrained/tokenizers/open_llama_7b' \
                --path_to_masked_model='roberta-base' \
                --path_to_masked_tokenizer='roberta-base' \
                --output_dir="./baseline_results" --suffix="gpt2-small_128_gutenberg_${chunk}_50neighbors_roberta" \
                --n_neighbors=50 --top_k=10
done

