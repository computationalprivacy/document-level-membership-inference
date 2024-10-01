from datasets import load_from_disk
from tqdm import tqdm
import random
import pickle
from collections import Counter
import argparse

def get_train_save_test(all_tokenized, n_docs_each, output_dir, dataset_name):
    indices = range(len(all_tokenized))
    train_sample_idx = random.sample(indices, n_docs_each)
    
    train_sample = all_tokenized.select(train_sample_idx)
    test_sample_idx = [idx for idx in indices if idx not in train_sample_idx]

    with open(f'{output_dir}/{dataset_name}_no_token_freq_indices.pickle', 'wb') as f:
        pickle.dump(test_sample_idx, f)

    return train_sample

def get_token_count(sample, token_count: dict = None):
    # Let's build up the dictionary of token count gradually
    if token_count is None:
        token_count = dict()
    
    # first do members
    for i, sample in tqdm(enumerate(sample)):
        doc_tokens = sample['input_ids']
        integer_count = Counter(doc_tokens)
        doc_dict = dict(integer_count)
    
        for key in doc_dict.keys():
            if key in token_count.keys():
                token_count[key] += doc_dict[key]
            else:
                token_count[key] = doc_dict[key]

    return token_count

def get_token_freq(token_count):
    total_sum = sum(val for val in token_count.values())
    print('Total sum: ', total_sum)
    
    token_freq = token_count.copy()
    
    for key in token_freq.keys():
        token_freq[key] = token_freq[key] / total_sum

    return token_freq

def get_token_freq_all(members_train_sample, non_members_train_sample):

    # first do members
    token_count = get_token_count(members_train_sample)
    
    # then do non members
    token_count = get_token_count(non_members_train_sample, token_count)
    
    token_freq = get_token_freq(token_count)

    return token_freq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_member", type=str, required=True)
    parser.add_argument("--path_to_non_member", type=str, required=True)
    parser.add_argument("--path_to_token_freq", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--n_docs_each", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    PATH_TO_MEMBER = args.path_to_member
    PATH_TO_NON_MEMBER = args.path_to_non_member
    N_DOCS_EACH = args.n_docs_each
    PATH_TO_TOKEN_FREQ = args.path_to_token_freq
    PREFIX = args.prefix

    random.seed(args.seed)
        
    print(f"Loading all data...")
    all_tokenized_member = load_from_disk(PATH_TO_MEMBER)
    all_tokenized_non_member = load_from_disk(PATH_TO_NON_MEMBER)

    print(f"Splitting the datasets...")
    members_train_sample = get_train_save_test(all_tokenized_member, N_DOCS_EACH, PATH_TO_TOKEN_FREQ, 
                                              PATH_TO_MEMBER.split('/')[-1])
    non_members_train_sample = get_train_save_test(all_tokenized_non_member, N_DOCS_EACH, PATH_TO_TOKEN_FREQ, 
                                              PATH_TO_NON_MEMBER.split('/')[-1])
    
    token_freq = get_token_freq(members_train_sample, non_members_train_sample)
    
    with open(f'{PATH_TO_TOKEN_FREQ}/token_frequency_{PREFIX}_{N_DOCS_EACH}.pickle', 'wb') as f:
        pickle.dump(token_freq, f)

if __name__ == "__main__":
    main()
