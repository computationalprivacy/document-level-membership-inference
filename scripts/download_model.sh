#!/bin/bash

cd /data_2/matthieu/copyright/GitFolder/copyright_inference/

python src/import_model.py --model="openlm-research/open_llama_7b" --write_dir="pretrained"
