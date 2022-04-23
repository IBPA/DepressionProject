# -*- coding: utf-8 -*-
"""run tsne and tsne for outliers

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * include reformat.
    * I don;t like preprocessor...
    * Help for clicks

python -u -m DepressionProjectNew.run_tsne_use_rfe_results \
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
    get_curve_metrics)
from msap.utils import (
    ClassifierHandler,
    load_X_and_y,
    dump_X_and_y,
    KFold_by_feature)
from msap.utils.plot import (
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
    rfe12 = ['a521r_8wg', 'd801_12wg', \
            'd602ar_12wg', 'Adaptability_24m', \
            'Avg_FinDiff_61m', 'f9ms012_96m', 'f9ms018_108m', \
            'fddp130_120m', 'fdms012_120m', 'fdms026_120m', \
            'Avg_neighb_122m', 'kz021_0m_1.0_0_2.0_1', \
            'married8wg_8wg_0.0_0_1.0_1', 'divorced8wg_8wg_0.0_0_1.0_1', \
            'separated8wg_8wg_0.0_0_1.0_1', 'c800_32wg_3.0', \
            'c800_32wg_6.0', 'c800_32wg_7.0', 'c800_32wg_9.0', \
            'c801_32wg_1.0', 'c801_32wg_7.0', 'c801_32wg_8.0', \
            'c801_32wg_9.0']
    rfe16 = ['kz021_0m_1.0_0_2.0_1']
    rfe17 = ['d781_12wg', 'Avg_sc_m_47m', 'Avg_neighb_m_122m', 'kz021_0m_1.0_0_2.0_1']
    rfe18 = ['Max_ed_32wg', 'Avg_FinDiff_61m', 'kz021_0m_1.0_0_2.0_1', 'f020a_8m_1.0_0_2.0_1']

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

    # tsne for best rfe results
    plot_tsne( # removed outliers
        X = X_12[rfe12],
        y = y_12,
        random_state = random_state,
        path_save = f"{path_data_dir_12}/"
        "tsne_from_rfe.png")
    
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
    y_12_pred_allpos = pd.Series(np.ones(len(y_12)))
    p_base = precision_score(y_12, y_12_pred_allpos)
    method = 'pr'
    try:
        curve_metrics = get_curve_metrics(
            clf, X_12, y_12, method, splits)
    except Exception as e:
        logger.info(
            f"{method} skipped due to data inbalance. Error Type: "
            f"{type(e)}. Error message: {e}")

    plot_curves(
        curve_metrics,
        method=method,
        pr_base = p_base,
        path_save=f"{path_data_dir_12}/{method}_fixed.png")
    

    

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
    # tsne for best rfe results
    plot_tsne( # removed outliers
        X = X_16[rfe16],
        y = y_16,
        random_state = random_state,
        path_save = f"{path_data_dir_16}/"
        "tsne_from_rfe.png")
    
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
    y_16_pred_allpos = pd.Series(np.ones(len(y_16)))
    p_base = precision_score(y_16, y_16_pred_allpos)
    method = 'pr'
    try:
        curve_metrics = get_curve_metrics(
            clf, X_16, y_16, method, splits)
    except Exception as e:
        logger.info(
            f"{method} skipped due to data inbalance. Error Type: "
            f"{type(e)}. Error message: {e}")

    plot_curves(
        curve_metrics,
        method=method,
        pr_base = p_base,
        path_save=f"{path_data_dir_16}/{method}_fixed.png")
    

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
    # tsne for best rfe results
    plot_tsne( # removed outliers
        X = X_17[rfe17],
        y = y_17,
        random_state = random_state,
        path_save = f"{path_data_dir_17}/"
        "tsne_from_rfe.png")

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
    y_17_pred_allpos = pd.Series(np.ones(len(y_17)))
    p_base = precision_score(y_17, y_17_pred_allpos)
    method = 'pr'
    try:
        curve_metrics = get_curve_metrics(
            clf, X_17, y_17, method, splits)
    except Exception as e:
        logger.info(
            f"{method} skipped due to data inbalance. Error Type: "
            f"{type(e)}. Error message: {e}")

    plot_curves(
        curve_metrics,
        method=method,
        pr_base = p_base,
        path_save=f"{path_data_dir_17}/{method}_fixed.png")

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
    # tsne for best rfe results
    plot_tsne( # removed outliers
        X = X_18[rfe18],
        y = y_18,
        random_state = random_state,
        path_save = f"{path_data_dir_18}/"
        "tsne_from_rfe.png")
    
    splits = KFold_by_feature(
        X=X_18,
        n_splits=5,
        feature=feature_kfold,
        random_state=random_state)
    if feature_kfold is not None:
        X_18 = X_18.drop([feature_kfold], axis=1)
    
    # replot pr curve but with correct baseline
    y_18_pred_allpos = pd.Series(np.ones(len(y_18)))
    p_base = precision_score(y_18, y_18_pred_allpos)
    method = 'pr'
    clf = ClassifierHandler(
        classifier_mode=best_clf_18,
        params=best_cv_result_18['param'],
        random_state=random_state).clf
    try:
        curve_metrics = get_curve_metrics(
            clf, X_18, y_18, method, splits)
    except Exception as e:
        logger.info(
            f"{method} skipped due to data inbalance. Error Type: "
            f"{type(e)}. Error message: {e}")

    plot_curves(
        curve_metrics,
        method=method,
        pr_base = p_base,
        path_save=f"{path_data_dir_18}/{method}_fixed.png")
    
    

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