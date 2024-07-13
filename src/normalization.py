import numpy as np
from tqdm import tqdm
from datasets import Dataset

def get_norm_val(token, token_norm_dict):
    if token in token_norm_dict.keys():
        return token_norm_dict[token]
    else:
        return min(token_norm_dict.values()) / 2.0

def get_diff_max_token(idd: int, dataset: Dataset,
                       probas: dict, nlls: dict, ranks: dict, token_norm: dict):
    # start simple: just take the diff between max and the token proba
    all_tokens = dataset[idd]['input_ids']
    diffs = []
    for token_idx in nlls.keys():
        max_proba = max(probas[token_idx].values())
        token_proba = np.exp(-nlls[token_idx])
        token = all_tokens[token_idx]
        token_norm_val = get_norm_val(token, token_norm)
        norm_diff = -np.log((1 - (max_proba - token_proba)) / token_norm_val)
        if np.isnan(norm_diff) or np.isinf(norm_diff):
            print(norm_diff)
            norm_diff = 10e-6
        diffs.append(norm_diff)

    return diffs

def normalize_token_loss(idd: int, dataset: Dataset,
                         nlls: dict, token_norm: dict):
    # let's grab the relevant tokens for the book id
    all_tokens = dataset[idd]['input_ids']

    normalized_nlls = []

    for token_idx in nlls.keys():
        token = all_tokens[token_idx]
        token_prob = np.exp(-nlls[token_idx])
        norm_loss = -np.log(token_prob / get_norm_val(token, token_norm))
        if np.isnan(norm_loss) or np.isinf(norm_loss):
            print(norm_loss)
            norm_loss = 10e-6
        normalized_nlls.append(norm_loss)

    return normalized_nlls

def normalize_tokens_doc(raw_values: dict, raw_dataset: Dataset,
                         token_norm: dict, norm_type: str = 'ratio'):

    all_normalized_values = dict()
    for idd in tqdm(raw_values.keys()):
        ppl, nlls, probas, ranks = raw_values[idd]
        if norm_type == 'ratio':
            normalized_values = normalize_token_loss(idd, dataset=raw_dataset,
                                                   nlls=nlls, token_norm=token_norm)
        elif norm_type == 'diff_max_token_proba':
            normalized_values = get_diff_max_token(idd, dataset=raw_dataset,
                                                 probas=probas, nlls=nlls, ranks=ranks, token_norm=token_norm)
        elif norm_type == 'none':
            normalized_values = list(nlls.values())
        else:
            print('No valid normalization type has been provided.')
            print(norm_type)
            raise Exception
        all_normalized_values[idd] = normalized_values

    return all_normalized_values