import numpy as np
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_from_disk
import random 
import argparse

from compute_token_freq import get_token_count, get_token_freq

parser = argparse.ArgumentParser()

parser.add_argument("--path_to_model", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--path_to_data_indx", type=str, default=None)
parser.add_argument("--data_indx_name", type=str, default='')
parser.add_argument("--path_to_tokenizer", type=str, required=True)
parser.add_argument("--path_to_dataset", required=True, type=str)
parser.add_argument("--results_dir", required=True, type=str)
parser.add_argument("--general_proba_path", type=str, default='general_proba.pickle')
parser.add_argument("--token_freq_path", type=str, default='token_freq.pickle')
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--stride", type=int, default=2048)
parser.add_argument("--top_probas", type=int, default=10)
parser.add_argument("--nb_samples", type=int, required=False)
parser.add_argument("--shuffle", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

def compute_perplexity(all_tokens_one_doc: list, model: LlamaForCausalLM,
                       stride: int, max_length: int, top_probas: int, device):
    
    seq_len = len(all_tokens_one_doc)
    nlls = dict()  # dict with key token idx in doc and val negative log likelihood 
    probas = dict() # dict with as key token idx in doc and as value the top predicted probas over the vocab with its token
    ranks = dict() # a dict with as key token idx in doc and as value the rank in the predicted probas associated with the correct token
    # note that this only works if you have at most one value per token. 
    prev_end_loc = 0

    # in order to compute the general proba, save all probas
    probas_sum = np.zeros(model.config.vocab_size)
    n_samples = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        input_ids =  torch.tensor(all_tokens_one_doc[begin_loc:end_loc]).reshape(1, -1).to(device) 
        target_ids = input_ids.clone().to(device)

        outputs = model(input_ids, labels=target_ids)

        # we shift the targets as such that we don't consider the first token as target, but we do consider all the rest
        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
        shift_targets = target_ids[..., 1:].contiguous().view(-1)

        # so we get a loss for max_length - 1 tokens, being for all tokens expect for the first one
        loss = F.cross_entropy(shift_logits, shift_targets.view(-1), reduction='none')
        loss_list = list(loss.detach().cpu().numpy())

        # convert to probas
        probabilities = F.softmax(shift_logits, dim=1)
        
        # so if we allocate the loss to the token level, we should start from begin_loc + 1
        for i, idx in enumerate(range(begin_loc + 1, end_loc)):
            nlls[idx] = loss_list[i]
            top_probs, top_indices = torch.topk(probabilities[i, :], top_probas, dim=0)
            top_probs_cpu, top_indices_cpu = top_probs.cpu().numpy(), top_indices.cpu().numpy()
            probas[idx] = {token: top_probs_cpu[i] for i, token in enumerate(top_indices_cpu)}
            ranks[idx] = None #(probabilities[i, :] > probabilities[i, shift_targets[i]]).sum().item()

            # add the probas
            probas_sum += probabilities[i, :].detach().cpu().numpy()
            n_samples += 1

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    mean_probas = probas_sum / n_samples    
    mean_nll = np.mean(list(nlls.values()))
    doc_perplexity = np.exp(mean_nll)

    return doc_perplexity, nlls, probas, ranks, mean_probas

def main():
    DATA_DIR = args.data_dir
    PATH_TO_MODEL = args.path_to_model
    MODEL_NAME = PATH_TO_MODEL.split('/')[-1]
    PATH_TO_DATASET = args.path_to_dataset
    DATASET_NAME = PATH_TO_DATASET.split('/')[-1]
    PATH_TO_DATA_INDX = args.path_to_data_indx
    DATA_INDX_NAME = args.data_indx_name
    PATH_TO_TOKENIZER = args.path_to_tokenizer
    TOKENIZER_NAME = PATH_TO_TOKENIZER.split('/')[-1]
    RESULTS_DIR = args.results_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    print(f"Loading {DATASET_NAME}...")
    
    tokenized_dataset = load_from_disk(PATH_TO_DATASET)
    # check if the data dir path is a hf dataset or a set of indices as subset
    if PATH_TO_DATA_INDX is not None:
        with open(PATH_TO_DATA_INDX, 'rb') as f:
            indices = pickle.load(f)
            tokenized_dataset = tokenized_dataset.select(indices)
            
    model = LlamaForCausalLM.from_pretrained(PATH_TO_MODEL, torch_dtype=torch.float16).to(device)
    max_length = args.max_length
    stride = args.stride
    top_probas = args.top_probas

    all_nlls = dict()
    nb_samples = len(tokenized_dataset) if args.nb_samples is None else args.nb_samples

    # for the general proba
    all_probas = np.zeros(model.config.vocab_size)
    n_docs = 0

    print(f"Computing perplexity...")
    with torch.no_grad():
        if args.shuffle:
            random.seed(args.seed)
            samples_idx = random.sample(range(len(tokenized_dataset)), min(nb_samples, len(tokenized_dataset)))
            for idx in tqdm(samples_idx):
                sample = tokenized_dataset[idx]
                sample_input_ids = sample["input_ids"]
                all_results = compute_perplexity(all_tokens_one_doc=sample_input_ids, model=model, 
                                                   stride=stride, max_length=max_length, 
                                                   top_probas=top_probas, device=device)
                all_nlls[idx] = all_results[:-1]
                all_probas += all_results[-1]
                n_docs += 1
                print(f"Perplexity of doc {idx} with {len(all_nlls[idx][1])} batches: {all_nlls[idx][0]}")

            print(f"Computing token frequency...")
            # let's also compute token frequency
            token_count = get_token_count(tokenized_dataset.select(samples_idx))
            token_freq = get_token_freq(token_count)
        else: 
            for idx, sample in enumerate(tqdm(tokenized_dataset, total=nb_samples)):
                if idx >= nb_samples:
                    break
                sample_input_ids = sample["input_ids"]
                all_results = compute_perplexity(all_tokens_one_doc=sample_input_ids, model=model, 
                                                   stride=stride, max_length=max_length, 
                                                   top_probas=top_probas, device=device)
                all_nlls[idx] = all_results[:-1]
                all_probas += all_results[-1]
                n_docs += 1
                print(f"Perplexity of doc {idx} with {len(all_nlls[idx][1])} batches: {all_nlls[idx][0]}")

            print(f"Computing token frequency...")
            # let's also compute token frequency
            token_count = get_token_count(tokenized_dataset.select(list(range(nb_samples))))
            token_freq = get_token_freq(token_count)

    # save the perplexity results
    file_name = f"{RESULTS_DIR}/perplexity_{MODEL_NAME}_{TOKENIZER_NAME}_{DATASET_NAME}_{DATA_INDX_NAME}_{nb_samples}_{max_length}_{stride}_seed{args.seed}.pickle"
    with open(file_name, 'wb') as f:
        pickle.dump(all_nlls, f)
    print(f'Results saved as {file_name}')

    # save the general probability
    mean_probas_all = all_probas / n_docs
    with open(args.general_proba_path, 'wb') as f:
        pickle.dump(mean_probas_all, f)

    # save the token freq
    with open(args.token_freq_path, 'wb') as f:
        pickle.dump(token_freq, f)
    
if __name__ == "__main__":
    main()
