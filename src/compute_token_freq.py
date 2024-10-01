from tqdm import tqdm
from collections import Counter

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

