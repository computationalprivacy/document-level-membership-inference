# Document-level membership inference for Large Language Models

Given black-box access to a pretrained large language model, can we predict whether a document has been part of its training dataset? 

This repo contains the source code to generate the results as published in the paper ["Did the Neurons Read your Book? Document-level Membership Inference for Large Language Models"](https://arxiv.org/pdf/2310.15007). 

## 1. Install environment

Follow these steps to install the correct python environment:
- `conda create --name doc_membership python=3.9`
- `conda activate doc_membership`
- `pip install -r requirements.txt`

## 2. Model setup

We now download the target model we consider. Use `python src/split_chunks.py` or `scripts/download_model.sh` to do so for the desired model on Hugging Face. In the paper we used [OpenLLaMA](https://huggingface.co/openlm-research/open_llama_7b).

## 2. Dataset setup

First and foremost, textual data should be collected and split in 'member' and 'non member' documents. In this project both books from Project Gutenberg and academic papers from ArXiv have been considered. 

To reproduce the data collection we rely on the data download and preprocess scripts provided by RedPajama (their first version, so now an older branch [here](https://github.com/togethercomputer/RedPajama-Data/tree/rp_v1)). More specifically, we applied the following strategy for both data sources:
- Books Project Gutenberg. 
    - Members: we just downloaded PG-19 from Hugging Face, as for instance [here](https://huggingface.co/datasets/deepmind/pg19).
    - Non-members: we used public code to scrape books from Project Gutenberg using [this code](https://github.com/kpully/gutenberg_scraper). You can find the scripts we utilized to do so in `data/raw_gutenberg/`. Note that the book index to start from was manually searched from [Project Gutenberg](https://www.gutenberg.org/). 
- Academic papers from ArXiv. 
    - Members: we download all `jsonl` files as provided by the V1 version of RedPajama. For all details see `data/raw_arxiv/`.
    - Non-members: we download all ArXiv papers at a small cost using the resources ArXiv provides [here](https://info.arxiv.org/help/bulk_data_s3.html) and the script to do so [here](https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/run_download.py).
    - All preprocessing for ArXiv has been done using [this script](https://github.com/togethercomputer/RedPajama-Data/blob/rp_v1/data_prep/arxiv/run_clean.py).

Next, we also tokenize the data using `python src/tokenize_data.py` or `scripts/tokenize_data.sh`.

Lastly, we create 'chunks' of documents, enabling us to run the entire pipeline multiple times (training on k-1 chunks and evaluating on the heldout chunk, repeating this k times.)
For this we use `python src/split_chunks.py -c config/SOME_CONFIG.ini` with the appropriate input arguments. 

## 4. Computing the perplexity for all chunks

We will now query the downloaded language model while running through each document, computing for each token its predicted probability and the top probabilities. 
For this we use `python src/compute_perplexity.py` with the appropriate input arguments as in `scripts/compute_perplexity.sh`. Using GPUs is recommended for this. 
The resulting token-level values are saved in `perplexity_results/`. 

At the same time, the general probability for each token and token frequency in the overall set of documents is computed and saved. 

## 5. Training and evaluating the meta-classifier for membership prediction

We run this with `python main.py -c config/SOME_CONFIG.ini`, where the exact setup should be specified in the config file (such as the path to perplexity results, the normalization type, meta-classifier type etc). 
The evaluation results are then saved in `classifier_results/`. The folder `./config/` contains all setups used to generate the results in the paper (for one dataset, i.e. books). 

## 6. Compute baselines

We also provided the code we used to compute the baselines. For this we use `python src/compute_baselines.py` with the appropriate input arguments as in `scripts/compute_baselines.sh`. Note that the code comes from Shi et al. [here](https://github.com/swj0419/detect-pretrain-code).

For the neighborhood baseline as introduced by Mattern et al., we adapt [their code](https://github.com/mireshghallah/neighborhood-curvature-mia) to `src/compute_baselines.py` and `scripts/compute_neighborhood_baselines.sh`. Note that its input requires a pickle file, but this could be easily adapted if needed. `

## 7. Citation

If you found this code helpful for your research, kindly cite our work: 

```
@article{meeus2023did,
  title={Did the neurons read your book? document-level membership inference for large language models},
  author={Meeus, Matthieu and Jain, Shubham and Rei, Marek and de Montjoye, Yves-Alexandre},
  journal={arXiv preprint arXiv:2310.15007},
  year={2023}
}
```