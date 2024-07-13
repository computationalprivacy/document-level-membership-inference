#!/bin/bash
  
cd /data_2/matthieu/copyright/GitFolder/copyright_inference/

python src/compute_token_freq.py --path_to_member='./data/tokenized/open_llama_7b/arxiv_redpajama' --path_to_non_member='./data/tokenized/open_llama_7b/arxiv_non_member' --path_to_token_freq='./data/token_freq' --prefix='arxiv' --n_docs_each=50000
