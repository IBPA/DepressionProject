# Mainly for testing random seed on just preprocessing
# -*- coding: utf-8 -*-
"""
Model evaluation script, but uses another results.pkl's best model to
    preprocess and run best model for different age
    assumes cleaned data is correct though

Authors:
    Fangzhou Li - fzli@ucdavis.edu

"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
import click

from msap.modeling.configs import (
    ModelSelectionConfig)
from msap.modeling.model_selection.preprocessing import Preprocessor
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
    dump_X_and_y,
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

    cfg_model = ModelSelectionConfig

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

    # X_raw, _ = load_X_and_y(path_input_data_raw, col_y=feature_label)

    #X, y = load_X_and_y(
    #    f"{path_input_preprocessed_data_dir}/"
    #    f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}.csv",
    #    col_y=feature_label)


    # idxes_outlier = np.loadtxt(
    #     f"{path_input_preprocessed_data_dir}/"
    #     f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}"
    #     "_outlier_indices.txt",
    #     delimiter='\n',
    #     dtype=int)

    # preprocess data
    filename_data_scale_impute = cfg_model.get_filename_scale_impute_data(
        best_scale_mode, best_impute_mode, best_outlier_mode)
    filename_data_prep = cfg_model.get_filename_preprocessed_data(
        best_scale_mode, best_impute_mode, best_outlier_mode)
    filename_outliers = cfg_model.get_filename_outliers(
        best_scale_mode, best_impute_mode, best_outlier_mode)
    
    data = pd.read_csv(path_input_data_raw)

    if feature_kfold is not None:
        data = data.set_index(feature_kfold)

    X = data.drop([feature_label], axis=1)
    y = data[feature_label]

    # test
    #X = X[:60]
    #y = y[:60]

    try:
        preprocessor = Preprocessor(
            best_scale_mode,
            best_impute_mode,
            best_outlier_mode,
            random_state,
            f"{path_input_preprocessed_data_dir}/"
            f"{filename_data_scale_impute}")
        X_prep, y_prep, idxs_outlier = preprocessor.preprocess(X, y)
        dump_X_and_y(
            X=X_prep
            if feature_kfold is None else X_prep.reset_index(),
            y=y_prep
            if feature_kfold is None else y_prep.reset_index(
                drop=True),
            path_output_data=f"{path_input_preprocessed_data_dir}/"
            f"{filename_data_prep}")
        np.savetxt(
            f"{path_input_preprocessed_data_dir}/{filename_outliers}",
            idxs_outlier,
            fmt='%d')
    except Exception as e:
        logging.info(f"Something happened during preprocessing {e}")
        pass

if __name__ == '__main__':
    main()