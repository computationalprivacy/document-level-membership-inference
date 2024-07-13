#!/bin/bash
  
cd /data_2/matthieu/copyright/GitFolder/copyright_inference/

python src/tokenize_data.py --data_dir='./data' --path_to_tokenizer='./pretrained/tokenizers/open_llama_7b' --path_to_dataset='./data/raw/gutenberg_non_member' --nb_workers=40
