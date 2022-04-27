# -*- coding: utf-8 -*-
"""run tsne and tsne for outliers

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * 

python -u -m DepressionProjectNew.run_f1_calcs_baseline_all \
./DepressionProjectNew/output/10MVIout/output_12_yesmental \
./DepressionProjectNew/output/10MVIout/output_16_yesmental \
./DepressionProjectNew/output/10MVIout/output_17_yesmental \
./DepressionProjectNew/output/10MVIout/output_18_yesmental \
y12CH_Dep_YN_144m \
y16CH_Dep_YN_192m \
y17CH_Dep_YN_204m \
y18CH_Dep_YN_216m
"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import click
from sklearn.metrics import precision_score

from msap.modeling.configs import (
    GridSearchConfig,
    ModelSelectionConfig)
from msap.modeling.model_selection.train import train_grid_search_cv, train_cv
from msap.modeling.model_selection.preprocessing import Preprocessor
from msap.modeling.model_evaluation.statistics import (
    get_baseline_testing_statistics,
    get_curve_metrics)
from msap.utils import (
    ClassifierHandler,
    load_X_and_y,
    dump_X_and_y,
    KFold_by_feature)
from msap.utils.plot import (
    plot_confusion_matrix,
    plot_tsne_outliers,
    plot_curves,
    plot_tsne)

os.environ["PYTHONWARNINGS"] = (
    "ignore::RuntimeWarning"
)

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.DEBUG)

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
    'path-data-dir-12',
    type=str)
@click.argument(
    'path-data-dir-16',
    type=str)
@click.argument(
    'path-data-dir-17',
    type=str)
@click.argument(
    'path-data-dir-18',
    type=str)
@click.argument(
    'feature-label-12',
    type=str)
@click.argument(
    'feature-label-16',
    type=str)
@click.argument(
    'feature-label-17',
    type=str)
@click.argument(
    'feature-label-18',
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
        path_data_dir_12,
        path_data_dir_16,
        path_data_dir_17,
        path_data_dir_18,
        feature_label_12,
        feature_label_16,
        feature_label_17,
        feature_label_18,
        feature_kfold,
        random_state):
    """
    """
    
    cfg_model = ModelSelectionConfig

    # Loading best model data from all ages
    # age 12
    model_selection_result_12 = None
    with open(f"{path_data_dir_12}/results.pkl", 'rb') as f:
        model_selection_result_12 = pickle.load(f)
    best_candidate_per_clf_12 = parse_model_selection_result(
        model_selection_result_12)
    best_candidate_12 = max(best_candidate_per_clf_12, key=lambda x: x[1])
    _, best_combination_12, best_cv_result_12 = best_candidate_12[0]
    best_scale_mode_12, best_impute_mode_12, best_outlier_mode_12, best_clf_12 \
        = best_combination_12
    X_12, y_12 = load_X_and_y(
        f"{path_data_dir_12}/preprocessed/"
        f"{best_scale_mode_12}_{best_impute_mode_12}_{best_outlier_mode_12}.csv",
        col_y=feature_label_12)

    splits = KFold_by_feature(
        X=X_12,
        n_splits=5,
        feature=feature_kfold,
        random_state=random_state)
    if feature_kfold is not None:
        X_12 = X_12.drop([feature_kfold], axis=1)
    
    # replot pr curve but with correct baseline
    clf = ClassifierHandler(
        classifier_mode=best_clf_12,
        params=best_cv_result_12['param'],
        random_state=random_state).clf
    
    # Plot confusion matrix for baseline
    baseline_12 = get_baseline_testing_statistics(
        clf, X_12, y_12, splits)
    plot_confusion_matrix(
        cv_result=baseline_12,
        axis_labels=['Depressed', 'Not Depressed'],
        path_save=f"{path_data_dir_12}/cm_baseline.png")


    # age 16
    model_selection_result_16 = None
    with open(f"{path_data_dir_16}/results.pkl", 'rb') as f:
        model_selection_result_16 = pickle.load(f)
    best_candidate_per_clf_16 = parse_model_selection_result(
        model_selection_result_16)
    best_candidate_16 = max(best_candidate_per_clf_16, key=lambda x: x[1])
    _, best_combination_16, best_cv_result_16 = best_candidate_16[0]
    best_scale_mode_16, best_impute_mode_16, best_outlier_mode_16, best_clf_16 \
        = best_combination_16
    X_16, y_16 = load_X_and_y(
        f"{path_data_dir_16}/preprocessed/"
        f"{best_scale_mode_16}_{best_impute_mode_16}_{best_outlier_mode_16}.csv",
        col_y=feature_label_16)
    
    splits = KFold_by_feature(
        X=X_16,
        n_splits=5,
        feature=feature_kfold,
        random_state=random_state)
    if feature_kfold is not None:
        X_16 = X_16.drop([feature_kfold], axis=1)
    
    # replot pr curve but with correct baseline
    clf = ClassifierHandler(
        classifier_mode=best_clf_16,
        params=best_cv_result_16['param'],
        random_state=random_state).clf
    
    # Plot confusion matrix for baseline
    baseline_16 = get_baseline_testing_statistics(
        clf, X_16, y_16, splits)
    plot_confusion_matrix(
        cv_result=baseline_16,
        axis_labels=['Depressed', 'Not Depressed'],
        path_save=f"{path_data_dir_16}/cm_baseline.png")
    

    # age 17
    model_selection_result_17 = None
    with open(f"{path_data_dir_17}/results.pkl", 'rb') as f:
        model_selection_result_17 = pickle.load(f)
    best_candidate_per_clf_17 = parse_model_selection_result(
        model_selection_result_17)
    best_candidate_17 = max(best_candidate_per_clf_17, key=lambda x: x[1])
    _, best_combination_17, best_cv_result_17 = best_candidate_17[0]
    best_scale_mode_17, best_impute_mode_17, best_outlier_mode_17, best_clf_17 \
        = best_combination_17
    X_17, y_17 = load_X_and_y(
        f"{path_data_dir_17}/preprocessed/"
        f"{best_scale_mode_17}_{best_impute_mode_17}_{best_outlier_mode_17}.csv",
        col_y=feature_label_17)

    splits = KFold_by_feature(
        X=X_17,
        n_splits=5,
        feature=feature_kfold,
        random_state=random_state)
    if feature_kfold is not None:
        X_17 = X_17.drop([feature_kfold], axis=1)

    # replot pr curve but with correct baseline
    clf = ClassifierHandler(
        classifier_mode=best_clf_17,
        params=best_cv_result_17['param'],
        random_state=random_state).clf
    
    # Plot confusion matrix for baseline
    baseline_17 = get_baseline_testing_statistics(
        clf, X_17, y_17, splits)
    plot_confusion_matrix(
        cv_result=baseline_17,
        axis_labels=['Depressed', 'Not Depressed'],
        path_save=f"{path_data_dir_17}/cm_baseline.png")
    

    # age 18
    model_selection_result_18 = None
    with open(f"{path_data_dir_18}/results.pkl", 'rb') as f:
        model_selection_result_18 = pickle.load(f)
    best_candidate_per_clf_18 = parse_model_selection_result(
        model_selection_result_18)
    best_candidate_18 = max(best_candidate_per_clf_18, key=lambda x: x[1])
    _, best_combination_18, best_cv_result_18 = best_candidate_18[0]
    best_scale_mode_18, best_impute_mode_18, best_outlier_mode_18, best_clf_18 \
        = best_combination_18
    X_18, y_18 = load_X_and_y(
        f"{path_data_dir_18}/preprocessed/"
        f"{best_scale_mode_18}_{best_impute_mode_18}_{best_outlier_mode_18}.csv",
        col_y=feature_label_18)
    
    splits = KFold_by_feature(
        X=X_18,
        n_splits=5,
        feature=feature_kfold,
        random_state=random_state)
    if feature_kfold is not None:
        X_18 = X_18.drop([feature_kfold], axis=1)
    
    clf = ClassifierHandler(
        classifier_mode=best_clf_18,
        params=best_cv_result_18['param'],
        random_state=random_state).clf
    
    # Plot confusion matrix for baseline
    baseline_18 = get_baseline_testing_statistics(
        clf, X_18, y_18, splits)
    print(baseline_18)
    plot_confusion_matrix(
        cv_result=baseline_18,
        axis_labels=['Depressed', 'Not Depressed'],
        path_save=f"{path_data_dir_18}/cm_baseline.png")
    

    #filename_data_prep = cfg_model.get_filename_preprocessed_data(
    #    scale_mode, impute_mode, outlier_mode)
    #data = pd.read_csv(f"{path_data_preprocessed_dir}/preprocessed/{filename_data_prep}")
    #X_prep = data.drop([feature_label], axis=1)
    #y_prep = data[feature_label]
    
    #plot_tsne( # removed outliers
    #    X = X_prep,
    #    y = y_prep,
    #    random_state = random_state,
    #    path_save = f"{path_data_preprocessed_dir}/"
    #    f"{filename_tsne}")

if __name__ == '__main__':
    main()