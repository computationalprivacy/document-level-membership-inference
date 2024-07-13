#!/bin/bash

for chunk in 0 1 2 3 4
do
    CUDA_VISIBLE_DEVICES=0 python src/compute_perplexity.py --data_dir='./data' \
                    --path_to_tokenizer='./pretrained/tokenizers/open_llama_7b' --path_to_model='./pretrained/models/open_llama_7b' \
                    --path_to_dataset="./data/final_chunks/gutenberg_${chunk}_min_tokens5000_seed42" \
                    --results_dir='./perplexity_results' --nb_samples=400 --stride=127 --max_length=128  \
                    --top_probas=10 --shuffle=0 \
                    --general_proba_path="./data/final_chunks/general_proba/general_proba_gutenberg_${chunk}_128.pickle" \
                    --token_freq_path="./data/final_chunks/token_freq/token_freq_gutenberg_${chunk}.pickle"
done