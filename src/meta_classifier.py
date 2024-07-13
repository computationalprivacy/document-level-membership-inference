import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

def split_data(member_vals: dict, non_member_vals: dict,
               test_size: float = 0.2, seed: int = 42) -> tuple:
    all_X = [member_vals[key] for key in member_vals.keys()] + [non_member_vals[key] for key in non_member_vals.keys()]
    all_y = [1] * len(member_vals) + [0] * len(non_member_vals)
    data_train, data_test, y_train, y_test = train_test_split(all_X, all_y,
                                                        test_size=test_size, random_state=seed)
    return data_train, data_test, y_train, y_test

def validate_clf(clf, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_test: pd.DataFrame) -> dict:
    y_train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print('Training accuracy: ', train_acc)
    train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    print('Training auc: ', train_auc)

    y_test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print('Test accuracy: ', test_acc)
    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print('Test auc: ', test_auc)
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])

    # let's add this as well
    train_pred_proba = clf.predict_proba(X_train)[:, 1]
    test_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    return {'train_acc': train_acc, 'train_auc': train_auc, 'test_acc':test_acc, 'test_auc': test_auc, 
           'test_fpr':fpr, 'test_tpr':tpr, 'test_thresholds': thresholds, 
            'train_pred_proba':train_pred_proba, 'test_pred_proba':test_pred_proba}

# let's normalize
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    for col in X_train.columns:
        minn = X_train[col].min()
        maxx = X_train[col].max()
        if maxx == minn:
            X_train[col] = 0.0
            X_test[col] = 0.0
        else:
            X_train[col] = (X_train[col] - minn) / (maxx - minn)
            X_test[col] = (X_test[col] - minn) / (maxx - minn)

    return X_train, X_test

def train_LogisticRegression(X_train: pd.DataFrame, y_train: pd.DataFrame) -> LogisticRegression:
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf

def train_RandomForest(X_train: pd.DataFrame, y_train: pd.DataFrame) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=500, max_depth=5, min_samples_leaf=3)
    clf.fit(X_train, y_train)
    return clf

def train_MLP(X_train: pd.DataFrame, y_train: pd.DataFrame) -> MLPClassifier:
    clf = MLPClassifier(hidden_layer_sizes=(100, ), alpha=0.01)
    clf.fit(X_train, y_train)
    return clf

def fit_validate_classifiers(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame,
                             y_test: pd.DataFrame, models: str) -> tuple:
    trained_models, all_results = [], dict()
    model_names = models.split(',')
    for model in model_names:
        print('Model: ', model)
        if model == 'logistic_regression':
            clf = train_LogisticRegression(X_train, y_train)
            results = validate_clf(clf, X_train, y_train, X_test,y_test)
        elif model == 'random_forest':
            clf = train_RandomForest(X_train, y_train)
            results = validate_clf(clf, X_train, y_train, X_test,y_test)
        elif model == 'mlp':
            clf = train_MLP(X_train, y_train)
            results = validate_clf(clf, X_train, y_train, X_test,y_test)
        else:
            print('Not a valid model.')
        print('---')
        trained_models.append(clf)
        all_results[model] = results

    return trained_models, all_results