# -*- coding: utf-8 -*-
"""Model evaluation script.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
import click
from imblearn.over_sampling import SMOTE

from msap.modeling.model_evaluation.statistics import (
    get_embedded_data,
    get_selected_features,
    get_curve_metrics,
    get_training_statistics,
    get_similarity_matrix)
from msap.explanatory_analysis import get_pairwise_correlation
from msap.utils import (
    ClassifierHandler,
    load_X_and_y,
    KFold_by_feature)
from msap.utils.plot import (
    plot_heatmap,
    plot_embedded_scatter,
    plot_rfe_line,
    plot_curves,
    plot_confusion_matrix)

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO)

METHODS_PC = ['pearson', 'spearman', 'kendall']
METHODS_EMBEDDING = ['tsne', 'pca']
METHODS_CURVE = ['pr', 'roc']
CLASSIFIER_MODES = [
    'decisiontreeclassifier',
    'gaussiannb',
    'multinomialnb',
    'svc',
    'adaboostclassifier',
    'randomforestclassifier',
    'mlpclassifier']


def parse_model_selection_result(ms_result: tuple) -> list:
    """Parse the model selection result tuple and get the best models.

    Args:
        ms_result: Model selection result tuple.

    Returns:
        List of best model and statistics for each classifiers.

    """
    candidates, _ = ms_result
    candidates = [(i, c, cv['best']) for i, c, cv in candidates]

    f1s_mean = []
    for i, c, cv_best in candidates:
        # Iterate over splits to calculate average F1 score.
        f1s = [cv_best[f'split_{j}']['f1'] for j in range(len(cv_best) - 1)]
        f1s_mean += [np.mean(np.nan_to_num(f1s))]

    candidates = list(zip(candidates, f1s_mean))
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    best_candidate_per_clf = []
    for clf in CLASSIFIER_MODES:
        for (i, c, cv_best), f1_mean in candidates:
            if c[3] == clf:
                if cv_best['param'] is not None:
                    cv_best['param'] = {k.split('__')[-1]: v
                                        for k, v in cv_best['param'].items()}

                best_candidate_per_clf += [((i, c, cv_best), f1_mean)]
                break

    return best_candidate_per_clf


@click.command()
@click.argument(
    'path-input-model-selection-result',
    type=click.Path(exists=True))
@click.argument(
    'path-input-preprocessed-data-dir',
    type=click.Path(exists=True))
@click.argument(
    'path-input-data-raw',
    type=click.Path(exists=True))
@click.argument(
    'path-output-dir',
    type=str)
@click.argument(
    'feature-label',
    type=str)
@click.option(
    '--feature-kfold',
    type=str,
    default=None)
@click.option(
    '--random-state',
    type=int,
    default=42)
def main(
        path_input_model_selection_result,
        path_input_preprocessed_data_dir,
        path_input_data_raw,
        path_output_dir,
        feature_label,
        feature_kfold,
        random_state):
    """
    """
    if not os.path.exists(path_output_dir):
        os.mkdir(path_output_dir)

    model_selection_result = None
    with open(path_input_model_selection_result, 'rb') as f:
        model_selection_result = pickle.load(f)

    best_candidate_per_clf = parse_model_selection_result(
        model_selection_result)
    best_candidate = max(best_candidate_per_clf, key=lambda x: x[1])
    _, best_combination, best_cv_result = best_candidate[0]
    best_scale_mode, best_impute_mode, best_outlier_mode, best_clf \
        = best_combination
    pd.DataFrame(best_candidate_per_clf).to_csv(
        f"{path_output_dir}/best_clfs.csv")

    X_raw, _ = load_X_and_y(
        path_input_data_raw,
        col_y=feature_label)

    X, y = load_X_and_y(
        f"{path_input_preprocessed_data_dir}/"
        f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}.csv",
        col_y=feature_label)
    # idxes_outlier = np.loadtxt(
    #     f"{path_input_preprocessed_data_dir}/"
    #     f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}"
    #     "_outlier_indices.txt",
    #     delimiter='\n',
    #     dtype=int)

    splits = KFold_by_feature(
        X=X,
        y=y,
        n_splits=5,
        feature=feature_kfold,
        random_state=random_state)
    if feature_kfold is not None:
        X = X.drop([feature_kfold], axis=1)

    clf = ClassifierHandler(
        classifier_mode=best_clf,
        params=best_cv_result['param'],
        random_state=random_state).clf

    smote = SMOTE(
        sampling_strategy='minority',
        n_jobs=1,
        random_state=42567)

    sex = "kz021_0m_1.0_0_2.0_1"
    F1_validation = []
    #logging.info(f"Splits: {len(splits)} SplitsInside: {len(splits[0])}")
    for i in range(len(splits)):
        logging.info(f"Split {i+1}:")
        train = splits[i][0]
        test = splits[i][1]
        #logging.info(f"Split {i+1} train:{train}")
        #logging.info(f"Split {i+1} test:{test}")
        X_train = X.iloc[train]
        y_train = y.iloc[train]

        # Originally, 1 is female. In robust, 0 is female, anything else is male.
        # found this out by comparing raw female and male vs robust female and male
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        X_smote_female = X_smote[X_smote[sex] == 0]
        X_smote_male = X_smote[~(X_smote[sex] == 0)]

        y_smote_female = y_smote.loc[X_smote_female.index.tolist()]
        y_smote_male = y_smote.loc[X_smote_male.index.tolist()]
        # print(y_smote_female.value_counts())
        # print(y_smote_male.value_counts())
        dep_female = len(y_smote_female[y_smote_female == 1])
        nodep_female = len(y_smote_female[y_smote_female == 0])
        dep_male = len(y_smote_male[y_smote_male == 1])
        nodep_male = len(y_smote_male[y_smote_male == 0])
        logging.info(f"Dep Female: {dep_female}")
        logging.info(f"NoDep Female: {nodep_female}")
        logging.info(f"Dep male: {dep_male}")
        logging.info(f"NoDep male: {nodep_male}")
        # f1 assume all male = dep = pos
        tp = dep_male
        fp = nodep_male
        tn = nodep_female
        fn = dep_female
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1_male_alldep = 2*precision*recall/(precision+recall)
        # f1 assume all female = dep = pos
        tp = dep_female
        fp = nodep_female
        tn = nodep_male
        fn = dep_male
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        F1_female_alldep = 2*precision*recall/(precision+recall)
        if F1_male_alldep > F1_female_alldep:
            logging.info(f"F1 Male All Dep: {F1_male_alldep}")
        else:
            logging.info(f"F1 Female All Dep: {F1_female_alldep}")

        X_test = X.iloc[test]
        y_test = y.iloc[test]
        X_test_female = X_test[X_test[sex] == 0]
        X_test_male = X_test[~(X_test[sex] == 0)]
        y_test_female = y_test.loc[X_test_female.index.tolist()]
        y_test_male = y_test.loc[X_test_male.index.tolist()]
        dep_female = len(y_test_female[y_test_female == 1])
        nodep_female = len(y_test_female[y_test_female == 0])
        dep_male = len(y_test_male[y_test_male == 1])
        nodep_male = len(y_test_male[y_test_male == 0])
        if F1_male_alldep > F1_female_alldep:
            # f1 assume all male = dep = pos
            tp = dep_male
            fp = nodep_male
            tn = nodep_female
            fn = dep_female
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            F1_male_alldep_test = 2*precision*recall/(precision+recall)
            logging.info(f"F1 Male All Dep test: {F1_male_alldep_test}")
            F1_validation += [F1_male_alldep_test]
        else:
            tp = dep_female
            fp = nodep_female
            tn = nodep_male
            fn = dep_male
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            F1_female_alldep_test = 2*precision*recall/(precision+recall)
            logging.info(f"F1 Female All Dep test: {F1_female_alldep_test}")
            F1_validation += [F1_female_alldep_test]
        # Dist of dep in M vs. dep in F.
        # Pick the one is better F1
        # Use this to evaluate on validation no smote
        # Calculate val F1 for 5 folds
        # Calculate average

        #logging.info(f"Train Depressed info: \n{y_train.value_counts()}")
        #logging.info(f"Train Sex info: \n{X_train[sex].value_counts()}")
        # within gender, which depressed/not depressed
        # calculate F1_male and F1_female (need precision and recall)

        # print(X_train[sex])
        # X_test = X.iloc[test]
        # y_test = y.iloc[test]
        #logging.info(f"Test Depressed info: \n{y_test.value_counts()}")
        #logging.info(f"Test Sex info: \n{X_test[sex].value_counts()}")
        # print(X_test[sex])
        # calculate which F1 based on which is better
    logging.info(f"Ave F1 test: {sum(F1_validation)/len(F1_validation)}")


if __name__ == '__main__':
    main()
