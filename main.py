"""Main file for experimentation."""
import configargparse
import pickle
from datasets import load_from_disk
from tqdm import tqdm

from src.normalization import normalize_tokens_doc
from src.feature_extraction import extract_features
from src.meta_classifier import scale_features, fit_validate_classifiers

def get_parser():
    """Return parser for the args."""
    parser = configargparse.ArgParser(
        description='Experiments copyright inference attacks')
    parser.add_argument(
        '-c', '--config', required=False, is_config_file=True,
        help='Config file path')

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_chunks", type=int, required=True)
    parser.add_argument("--path_to_raw_data", type=str, required=True) # '/data/final_chunks/arxiv_XX_min_tokens5000_seed42'
    parser.add_argument("--path_to_labels", type=str, required=True) # '/data/final_chunks/arxiv_0_labels.pickle'
    parser.add_argument("--path_to_perplexity_results", type=str, required=True)
    # 'perplexity_open_llama_7b_open_llama_7b_arxiv_XX_min_tokens5000_seed42__400_2048_2047_seed42.pickle'
    parser.add_argument("--path_to_normalization_dict", type=str, required=True) 
    # '/data/final_chunks/general_proba/general_proba_arxiv_0_128.pickle'
    parser.add_argument("--norm_type", type=str, default='ratio')
    parser.add_argument("--feat_extraction_type", type=str, default='simple_agg')
    parser.add_argument("--models", type=str, default='logistic_regression,random_forest')
    parser.add_argument("--seed", type=int, default=42)

    return parser

def get_train_token_norm(chunks: list, path_to_raw_data: str, path_to_normalization_dict: str) -> dict:
    tokens_per_chunk = []
    print('Counting all tokens...')
    for chunk in chunks:
        raw_dataset = load_from_disk(path_to_raw_data.replace('XX', str(chunk)))
        n_tokens = sum([len(doc['input_ids']) for doc in raw_dataset])
        tokens_per_chunk.append(n_tokens)

    total_n_tokens = sum(tokens_per_chunk)

    print('Combining all norm dicts...')
    for i, chunk in enumerate(chunks):
        with open(path_to_normalization_dict.replace('XX', str(chunk)), 'rb') as f:
            token_norm = pickle.load(f)
            # if it's general proba we need to convert array to dict
            if not isinstance(token_norm, dict):
                token_norm = {i: token_norm[i] for i in range(len(token_norm))}
            for key in token_norm.keys():
                token_norm[key] = token_norm[key] * tokens_per_chunk[i] / total_n_tokens
        if i == 0:
            all_token_norm = token_norm.copy()
        else:
            for key in token_norm.keys():
                if key not in all_token_norm.keys():
                    all_token_norm[key] = token_norm[key]
                else:
                    all_token_norm[key] += token_norm[key]
                    
    print('Sum of all combined values: ', sum(all_token_norm.values()))

    return all_token_norm    

def prep_one_chunk(path_to_raw_data: str, path_to_perplexity_results: str, 
                token_norm: dict, norm_type: str, path_to_labels: str) -> tuple:
    print("Loading the raw data..")
    raw_dataset =  load_from_disk(path_to_raw_data)

    print("Loading the perplexity results...")
    with open(path_to_perplexity_results, 'rb') as f:
        perplex_results = pickle.load(f)

    print("Running the normalization...")
    perplex_results_normalized = normalize_tokens_doc(raw_values=perplex_results, raw_dataset=raw_dataset,
                                                     token_norm=token_norm, norm_type=norm_type)

    print("Loading the labels...")
    with open(path_to_labels, 'rb') as f:
        labels = pickle.load(f)
        if len(perplex_results_normalized) != len(labels):
            labels = labels[:len(perplex_results_normalized)]

    return perplex_results_normalized, labels

def main(args):
    """Main function to call."""

    results_per_fold = dict()
    
    for i in tqdm(range(args.n_chunks)):
        print(f'Starting on fold {i}..')
        test_chunk = i
        train_chunks = [j for j in range(args.n_chunks) if j != i]

        train_token_norm = get_train_token_norm(chunks=train_chunks, path_to_raw_data=args.path_to_raw_data, 
                                               path_to_normalization_dict=args.path_to_normalization_dict)

        train_perplex, train_labels = [], []
        for j in train_chunks:
            perplex_results_normalized, labels = prep_one_chunk(path_to_raw_data=args.path_to_raw_data.replace('XX', str(j)), 
                                                                path_to_perplexity_results=args.path_to_perplexity_results.replace('XX', str(j)),
                                                                path_to_labels=args.path_to_labels.replace('XX', str(j)),
                                                                token_norm=train_token_norm, norm_type=args.norm_type)
            train_perplex += [perplex_results_normalized[key] for key in perplex_results_normalized.keys()]
            train_labels += labels

        # now do the test
        test_perplex_results_normalized, test_labels = prep_one_chunk(path_to_raw_data=args.path_to_raw_data.replace('XX', str(test_chunk)), 
                                                        path_to_perplexity_results=args.path_to_perplexity_results.replace('XX', str(test_chunk)),
                                                        path_to_labels=args.path_to_labels.replace('XX', str(test_chunk)),
                                                        token_norm=train_token_norm, norm_type=args.norm_type)
        test_perplex = [test_perplex_results_normalized[key] for key in test_perplex_results_normalized.keys()]
        
        print("Extract features...")
        X_train, X_test = extract_features(train_perplex, test_perplex, type=args.feat_extraction_type)
    
        print("Train and validate the meta-classifier...")
        X_train, X_test = scale_features(X_train, X_test)
        trained_models, all_results = fit_validate_classifiers(X_train=X_train, y_train=train_labels,
                                                               X_test=X_test, y_test=test_labels,
                                                               models=args.models)

        results_per_fold[i] = all_results

    print("Saving results...")
    results_dict = vars(args)
    results_dict['results_per_fold'] = results_per_fold
    
    with open(f"{args.output_dir}/{args.experiment_name}.pickle", 'wb') as f:
        pickle.dump(results_dict, f)

if __name__ == '__main__':
    main(get_parser().parse_args())