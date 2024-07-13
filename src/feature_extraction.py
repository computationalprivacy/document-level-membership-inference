from tqdm import tqdm
import numpy as np
import pandas as pd

def simple_agg_feat_extraction(data: list,
                               percentiles: tuple = (1, 5, 10, 25, 50, 75, 90, 95, 99)) -> pd.DataFrame:
    X = []

    for nlls in tqdm(data):
        X_val = []

        X_val.append(np.mean(nlls))
        X_val.append(np.min(nlls))
        X_val.append(np.max(nlls))
        X_val.append(np.std(nlls))
        X_val.append(len(nlls))
        for perc in percentiles:
            X_val.append(np.percentile(nlls, perc))
        X.append(X_val)

    X_df = pd.DataFrame(np.array(X), columns = ['mean', 'min', 'max', 'std', 'count'] +
                                          ['perc_' + str(perc) for perc in percentiles])

    return X_df

def histogram_feats(data: list, bins = None, n_bins=100) -> tuple:

    # if bins is None (train), get all the bins
    if bins is None:
        all_entropy = []
        for nlls in data:
            all_entropy += nlls
        bins = np.histogram(all_entropy, bins=n_bins)[1]

    X = []

    for nlls in data:
        count_per_bin = np.histogram(nlls, bins, density=True)[0]
        X.append(count_per_bin)

    X_df = pd.DataFrame(np.array(X), columns = ['rel_count_bin_' + str(k) for k in range(n_bins)])

    return X_df, bins

def extract_features(data_train: list, data_test: list,
                     type: str = 'simple_agg') -> tuple:
    if type == 'simple_agg':
        X_train = simple_agg_feat_extraction(data_train)
        X_test = simple_agg_feat_extraction(data_test)
    elif type.split('_')[0] == 'hist':
        X_train, bins = histogram_feats(data_train, n_bins = int(type.split('_')[1]))
        X_test, _ = histogram_feats(data_test, bins=bins, n_bins = int(type.split('_')[1]))
    else:
        print("No valid feature extraction has been provided.")
        raise Exception
    return X_train, X_test