## doc to compute baselines
import os
import pickle
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import zlib
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--path_to_tokenized_data", type=str, required=True)
parser.add_argument("--path_to_target_model", type=str, required=True)
parser.add_argument("--path_to_target_tokenizer", type=str, required=True)
parser.add_argument("--path_to_reference_model", type=str, required=True)
parser.add_argument("--path_to_reference_tokenizer", type=str, required=True)

parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--suffix", type=str, required=True)
parser.add_argument("--seq_length", type=int, default=128)
parser.add_argument("--stride", type=int, default=127)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

def calculatePerplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)

    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
        
    return torch.exp(loss).item(), all_prob, loss.item()

def compute_baselines(text, model1, model2, tokenizer1, tokenizer2):
    
    baseline_scores = {}

    p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
    p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

    p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity(text, model2, tokenizer2, gpu=model2.device)
   
   # ppl
    baseline_scores["ppl"] = p1
    # Ratio of log ppl of large and small models
    baseline_scores["ppl/Ref_ppl (calibrate PPL to the reference model)"] = p1_likelihood-p_ref_likelihood

    # Ratio of log ppl of lower-case and normal-case
    baseline_scores["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    
    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    baseline_scores["ppl/zlib"] = np.log(p1)/zlib_entropy
    
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        baseline_scores[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()
    
    return baseline_scores

def main():
    
    seq_length = args.seq_length
    stride = args.stride

    # load data
    data = load_from_disk(args.path_to_tokenized_data)

    # load models
    print("Loading the models...")
    target_model = AutoModelForCausalLM.from_pretrained(args.path_to_target_model, torch_dtype=torch.float16).to("cuda:0")
    target_tokenizer = AutoTokenizer.from_pretrained(args.path_to_target_tokenizer, torch_dtype=torch.float16)
    reference_model = AutoModelForCausalLM.from_pretrained(args.path_to_reference_model, torch_dtype=torch.float16).to("cuda:1")
    reference_tokenizer = AutoTokenizer.from_pretrained(args.path_to_reference_tokenizer, torch_dtype=torch.float16)

    print("Getting the sequences...")
    doc_ids = list()
    all_seqs = list()
    for i in tqdm(range(len(data))):
        all_tokens_one_doc = data[i]["input_ids"]
        book_len = len(all_tokens_one_doc)
        prev_end_loc = 0
        for begin_loc in range(0, book_len, stride):
            end_loc = min(begin_loc + seq_length, book_len)
            trg_len = end_loc - prev_end_loc

            input_ids =  all_tokens_one_doc[begin_loc:end_loc]
            all_seqs.append(input_ids)
            doc_ids.append('doc_' + str(i))

    print("Computing baselines...")
    baseline_results = list()
    for tokens in tqdm(all_seqs):
        text = target_tokenizer.decode(tokens)
        result = compute_baselines(text, target_model, reference_model, target_tokenizer, reference_tokenizer)
        baseline_results.append(result)

    data={"doc_id": doc_ids, "sequence": all_seqs}
    for key in baseline_results[0].keys():
        data[key] = [seq_results[key] for seq_results in baseline_results]
    
    df = pd.DataFrame(data=data)
    with open(f"{args.output_dir}/baseline_results_{args.suffix}.csv", 'wb') as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    main()
