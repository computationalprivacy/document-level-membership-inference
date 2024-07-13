## doc to compute neightborhood baseline
## from https://arxiv.org/pdf/2305.18462
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaForMaskedLM, RobertaTokenizer
from heapq import nlargest
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--path_to_balanced_data", type=str, required=True)
parser.add_argument("--path_to_target_model", type=str, required=True)
parser.add_argument("--path_to_target_tokenizer", type=str, required=True)
parser.add_argument("--path_to_masked_model", type=str, default='roberta-base')
parser.add_argument("--path_to_masked_tokenizer", type=str, default='roberta-base')

parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--suffix", type=str, required=True)
parser.add_argument("--n_neighbors", type=int, default=50)
parser.add_argument("--top_k", type=int, default=10)

args = parser.parse_args()

# load models
print("Loading the models...")
attack_model = AutoModelForCausalLM.from_pretrained(args.path_to_target_model, torch_dtype=torch.float16).to("cuda:0")
attack_tokenizer = AutoTokenizer.from_pretrained(args.path_to_target_tokenizer, torch_dtype=torch.float16, legacy = False)
attack_tokenizer.pad_token = attack_tokenizer.eos_token
search_model = RobertaForMaskedLM.from_pretrained(args.path_to_masked_model).to("cuda:1")
search_tokenizer = RobertaTokenizer.from_pretrained(args.path_to_masked_tokenizer)
token_dropout = torch.nn.Dropout(p=0.7)

def generate_neighbors(text, num_word_changes=1, n_neighbors=args.n_neighbors, top_k=args.top_k):
    
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, 
                                      max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:

        target_token = text_tokenized[0, target_token_index]
        # model == 'roberta'
        embeds = search_model.roberta.embeddings(text_tokenized)
            
        embeds = torch.cat((embeds[:,:target_token_index,:], 
                            token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), 
                            embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], top_k, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):
            if not cand == target_token:
                if original_prob.item() == 1:
                    print("probability is one!")
                    replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                else:
                    replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())

    highest_scored = nlargest(n_neighbors, replacements, key=replacements.get)

    texts = []
    for single in highest_scored:
        alt = text_tokenized
        target_token_index, cand = single
        alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to('cuda:1'), 
                         alt[:,target_token_index+1:]), dim=1)
        alt_text = search_tokenizer.batch_decode(alt)[0]
        texts.append((alt_text, replacements[single]))

    return texts

def get_logprob_batch(text):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, 
                                      max_length = 512, return_tensors='pt').input_ids.to('cuda:0')

    ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=attack_tokenizer.pad_token_id)
    logits = attack_model(text_tokenized, labels=text_tokenized).logits[:,:-1,:].transpose(1, 2)
    manual_logprob = - ce_loss(logits, text_tokenized[:,1:])
    mask = manual_logprob!=0
    manual_logprob_means = (manual_logprob*mask).sum(dim=1)/mask.sum(dim=1)

    return manual_logprob_means.tolist()

def get_score(text):

    # tokenize the text using MLM
    tok_orig = search_tokenizer(text, padding = True, truncation = True, 
                                max_length = 512, return_tensors='pt').input_ids.to('cuda:1')
    
    orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ")

    original_score = get_logprob_batch(orig_dec)[0]

    with torch.no_grad():
        
        neighbors = generate_neighbors(text)

        neighbors_texts = [n[0].replace(" [SEP]", " ").replace("[CLS] ", " ") for n in neighbors]
        neighbor_scores = get_logprob_batch(neighbors_texts)
        
    final_score = original_score - np.mean(neighbor_scores)
    
    return final_score

def main():
    
    # load data
    with open(args.path_to_balanced_data, 'rb') as f:
        balanced_df = pickle.load(f)
    
    print("Computing baselines...")
    neighborhood_scores = list()
    for tokens in tqdm(balanced_df.sequence.values):
        text = attack_tokenizer.decode(tokens)
        score = get_score(text)
        neighborhood_scores.append(score)

    balanced_df['neighborhood_score'] = neighborhood_scores

    with open(f"{args.output_dir}/neighborhood_results_{args.suffix}.csv", 'wb') as f:
        pickle.dump(balanced_df, f)

if __name__ == "__main__":
    main()
