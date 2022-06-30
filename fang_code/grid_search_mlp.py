"""
Conclusion:
    -
"""
from time import time
import pickle

from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

from .metrics import get_metrics

METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'auc',
    'ap',
    'specificity',
    'balanced_accuracy',
    'tn',
    'fp',
    'fn',
    'tp',
]


def get_metrics_scorer(clf, X, y):
    y_score = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)

    return get_metrics(y, y_pred, y_score)


def parse_grid_search_cv_results(cv_results):
    results = []
    for metric in METRICS:
        for subset in ['train', 'test']:
            results += [
                [f"{metric}_{subset}"]
                + list(cv_results[f"mean_{subset}_{metric}"])
            ]
    results = pd.DataFrame(
        results,
        columns=[str(x) for x in ['metric'] + cv_results['params']]
    ).set_index('metric').T

    return results


if __name__ == '__main__':
    data = pd.read_csv(
        "./output/pval_filter_60_MVI/output_12to18_yesmental/preprocessed/standard_iterative_none.csv"
    )
    print(data.shape)
    print(data['y12to18_Dep_YN_216m'].value_counts())

    X = data.drop(['y12to18_Dep_YN_216m'], axis=1)
    y = data['y12to18_Dep_YN_216m']
    splits = list(
        StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        ).split(X, y)
    )

    # num_alpha_start =
    # num_alpha_end =
    # num_alpha_increment =
    num_max_iter_start = 100
    num_max_iter_end = 1001
    num_max_iter_increment = 200
    num_hidden_layers_start = 1
    num_hidden_layers_end = 3
    num_hidden_layers_increment = 1
    num_hidden_neurons_start = 10
    num_hidden_neurons_end = 81
    num_hidden_neurons_increment = 20
    param_grid = {
        'alpha':  np.logspace(1.16, 2, 5).tolist(),
        'max_iter': list(
            range(num_max_iter_start, num_max_iter_end,
                  num_max_iter_increment)),
        'hidden_layer_sizes':
            [(nn,) * nl
                for nn in range(num_hidden_neurons_start,
                                num_hidden_neurons_end,
                                num_hidden_neurons_increment)
                for nl in range(num_hidden_layers_start,
                                num_hidden_layers_end,
                                num_hidden_layers_increment)],
        'random_state': [42]
    }

    # param_grid = {
    #     'max_depth': [1, 2],
    #     'random_state': [42],
    #     'n_jobs': [1],
    # }
    st = time()
    gs_no_smote = GridSearchCV(
        estimator=MLPClassifier(),
        param_grid=param_grid,
        scoring=get_metrics_scorer,
        refit=False,
        n_jobs=-1,
        cv=splits,
        return_train_score=True,
        verbose=2
    )
    gs_no_smote.fit(X, y)

    cv_results_no_smote = gs_no_smote.cv_results_
    with open("./fang_code/outputs/grid_search/mlp/no_smote.pkl", 'wb') as f:
        pickle.dump(cv_results_no_smote, f)
    parse_grid_search_cv_results(cv_results_no_smote).to_csv(
        "./fang_code/outputs/grid_search/mlp/no_smote.csv"
    )

    # With SMOTE.
    sm = SMOTE(random_state=46843, n_jobs=1)
    pipeline = Pipeline(
        steps=[('smote', sm), ('clf', MLPClassifier())]
    )
    gs_smote = GridSearchCV(
        estimator=pipeline,
        param_grid={f"clf__{k}": v for k, v in param_grid.items()},
        scoring=get_metrics_scorer,
        refit=False,
        n_jobs=-1,
        cv=splits,
        return_train_score=True,
        verbose=2
    )
    gs_smote.fit(X, y)
    cv_results_smote = gs_smote.cv_results_
    with open("./fang_code/outputs/grid_search/mlp/smote.pkl", 'wb') as f:
        pickle.dump(cv_results_smote, f)
    parse_grid_search_cv_results(cv_results_smote).to_csv(
        "./fang_code/outputs/grid_search/mlp/smote.csv"
    )
    print(time() - st)
