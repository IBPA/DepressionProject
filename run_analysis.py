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
from typing import Union
from sklearn.metrics import precision_score
import sklearn

from msap.modeling.model_evaluation.statistics import (
    get_embedded_data,
    get_selected_features,
    get_curve_metrics,
    get_curve_metrics_test,
    get_training_statistics,
    get_baseline_training_statistics,
    get_validation_statistics,
    get_baseline_validation_statistics,
    get_testing_statistics,
    get_baseline_testing_statistics,
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
    plot_rfe_line_detailed,
    plot_curves,
    plot_confusion_matrix)
from msap.modeling.configs import (
    ModelSelectionConfig)

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


def plot_all_curves(
        clf: sklearn.base.BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        path_output_dir: str,
        use_smote_first: bool,  # name includes smote if smote first
        use_rfe: bool,
        splits: Union[int, list[list, list]] = None):
    # for baseline precision, predict all positive/depressed
    y_pred_allpos = pd.Series(np.ones(len(y_train)))
    p_base = precision_score(y_train, y_pred_allpos)

    # Calculate and plot curves
    for method in METHODS_CURVE:
        try:
            curve_metrics = get_curve_metrics(
                clf, X_train, y_train, method, splits)
        except Exception as e:
            logger.info(
                f"{method} skipped due to data inbalance. Error Type: "
                f"{type(e)}. Error message: {e}")
            continue

        if use_smote_first and use_rfe:
            filename = f"{method}_smote_rfe_val.svg"
        elif use_smote_first and not use_rfe:
            filename = f"{method}_smote_val.svg"
        elif not use_smote_first and use_rfe:
            filename = f"{method}_rfe_val.svg"
        else:
            filename = f"{method}_val.svg"
        plot_curves(
            curve_metrics,
            method=method,
            pr_base=p_base,
            path_save=f"{path_output_dir}/{filename}")

    y_pred_allpos = pd.Series(np.ones(len(y_test)))
    p_base = precision_score(y_test, y_pred_allpos)
    # Calculate and plot curves for test data
    for method in METHODS_CURVE:
        try:
            curve_metrics = get_curve_metrics_test(
                clf, X_train, y_train, X_test, y_test, method, splits)
        except Exception as e:
            logger.info(
                f"{method} skipped due to data inbalance. Error Type: "
                f"{type(e)}. Error message: {e}")
            continue

        if use_smote_first and use_rfe:
            filename = f"{method}_smote_rfe_test.svg"
        elif use_smote_first and not use_rfe:
            filename = f"{method}_smote_test.svg"
        elif not use_smote_first and use_rfe:
            filename = f"{method}_rfe_test.svg"
        else:
            filename = f"{method}_test.svg"
        plot_curves(
            curve_metrics,
            method=method,
            pr_base=p_base,
            path_save=f"{path_output_dir}/{filename}")


def plot_all_confusion_matrices(
        clf: sklearn.base.BaseEstimator,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        path_output_dir: str,
        use_smote_first: bool,
        use_rfe: bool,
        use_f1: bool,
        splits: Union[int, list[list, list]] = None,
        classifier: str = None):
    # for baseline, predict all positive/depressed
    if use_smote_first and use_rfe:
        fileprefix = "cm_smote_rfe"
    elif use_smote_first and not use_rfe:
        fileprefix = "cm_smote"
    elif not use_smote_first and use_rfe:
        fileprefix = "cm_rfe"
    else:
        fileprefix = "cm"
    if classifier is not None:
        fileprefix = "_".join([classifier, fileprefix])

    if use_f1:
        mode = "f1"
    else:
        mode = "balanced_accuracy"

    best_cv_result = get_validation_statistics(clf, X_train, y_train, splits)
    plot_confusion_matrix(
        cv_result=best_cv_result,
        axis_labels=['Depressed', 'Not Depressed'],
        mode=mode,
        path_save=f"{path_output_dir}/{fileprefix}_val.svg")

    best_cv_result_val_baseline = get_baseline_validation_statistics(
        clf, X_train, y_train, splits)
    plot_confusion_matrix(
        cv_result=best_cv_result_val_baseline,
        axis_labels=['Depressed', 'Not Depressed'],
        mode=mode,
        path_save=f"{path_output_dir}/{fileprefix}_val_baseline.svg")

    # Plot confusion matrix with various metrics for training.
    best_cv_result_train = get_training_statistics(
        clf, X_train, y_train, splits)
    plot_confusion_matrix(
        cv_result=best_cv_result_train,
        axis_labels=['Depressed', 'Not Depressed'],
        mode=mode,
        path_save=f"{path_output_dir}/{fileprefix}_train.svg")

    best_cv_result_train_baseline = get_baseline_training_statistics(
        clf, X_train, y_train, splits)
    plot_confusion_matrix(
        cv_result=best_cv_result_train_baseline,
        axis_labels=['Depressed', 'Not Depressed'],
        mode=mode,
        path_save=f"{path_output_dir}/{fileprefix}_train_baseline.svg")

    # Plot confusion matrix with various metrics for testing.
    best_cv_result_test = get_testing_statistics(
        clf, X_train, y_train, X_test, y_test, splits)
    plot_confusion_matrix(
        cv_result=best_cv_result_test,
        axis_labels=['Depressed', 'Not Depressed'],
        mode=mode,
        path_save=f"{path_output_dir}/{fileprefix}_test.svg")

    # Plot confusion matrix with various metrics for baseline testing.
    cv_result_test_baseline = get_baseline_testing_statistics(
        clf, X_train, y_train, X_test, y_test, splits)
    plot_confusion_matrix(
        cv_result=cv_result_test_baseline,
        axis_labels=['Depressed', 'Not Depressed'],
        mode=mode,
        path_save=f"{path_output_dir}/{fileprefix}_baseline_test.svg")


def plot_all_embeddings(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        path_output_dir: str,
        random_state: int,
        use_smote_first: bool,
        use_rfe: bool):
    # Plot embedded data points.
    y_scatter = y_train.map({1.0: 'Depressed', 0.0: 'Not Depressed'})
    y_scatter.name = 'Translation'
    for method in METHODS_EMBEDDING:
        X_embedded = pd.DataFrame(
            get_embedded_data(
                X_train,
                method=method, random_state=random_state))
        X_embedded.columns = ['First Dimension', 'Second Dimension']
        if use_smote_first and use_rfe:
            path = f"{path_output_dir}/embed_{method}_smote_rfe_train.svg"
        elif use_smote_first and not use_rfe:
            path = f"{path_output_dir}/embed_{method}_smote_train.svg"
        elif not use_smote_first and use_rfe:
            path = f"{path_output_dir}/embed_{method}_rfe_train.svg"
        else:
            path = f"{path_output_dir}/embed_{method}_train.svg"
        plot_embedded_scatter(
            X_embedded,
            y_scatter,
            title=f"{method.upper()}",
            path_save=path)
    return


def plot_all_correlations(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_label: str,
        path_output_dir: str,
        use_smote_first: bool,
        use_rfe: bool):
    # plot correlations
    for method in METHODS_PC:
        corr, pval = get_pairwise_correlation(
            X_train, y_train, method=method)
        y_corr = corr[feature_label].drop([feature_label])
        y_pval = pval[feature_label].drop([feature_label])
        idxes_rank = y_corr.abs().argsort().tolist()[::-1]

        rank = pd.concat(
            [y_corr[idxes_rank], y_pval[idxes_rank]],
            axis=1)
        rank.columns = ['corr', 'p-value']
        if use_smote_first and use_rfe:
            path = f"{path_output_dir}/pc_rank_{method}_smote_rfe_train.csv"
        elif use_smote_first and not use_rfe:
            path = f"{path_output_dir}/pc_rank_{method}_smote_train.csv"
        elif not use_smote_first and use_rfe:
            path = f"{path_output_dir}/pc_rank_{method}_rfe_train.csv"
        else:
            path = f"{path_output_dir}/pc_rank_{method}_train.csv"
        rank.to_csv(path)

        if use_smote_first and use_rfe:
            path = f"{path_output_dir}/pc_{method}_smote_rfe_train.png"
        elif use_smote_first and not use_rfe:
            path = f"{path_output_dir}/pc_{method}_smote_train.png"
        elif not use_smote_first and use_rfe:
            path = f"{path_output_dir}/pc_{method}_rfe_train.png"
        else:
            path = f"{path_output_dir}/pc_{method}_train.png"
        plot_heatmap(
            corr,
            title=f"Pairwise {method.capitalize()} Correlation",
            path_save=path)
    return


def plot_similarity_matrix(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        path_output_dir: str,
        use_smote_first: bool,
        use_rfe: bool):
    sm = get_similarity_matrix(X_train, y_train)
    if use_smote_first and use_rfe:
        path = f"{path_output_dir}/sim_smote_rfe_train.png"
    elif use_smote_first and not use_rfe:
        path = f"{path_output_dir}/sim_smote_train.png"
    elif not use_smote_first and use_rfe:
        path = f"{path_output_dir}/sim_rfe_train.png"
    else:
        path = f"{path_output_dir}/sim_train.png"
    plot_heatmap(
        sm,
        title=f"Similarity Matrix",
        cmap='Greys',
        path_save=path)
    return


def parse_model_selection_result(ms_result: tuple, mode: str) -> list:
    """Parse the model selection result tuple and get the best models.

    Args:
        ms_result: Model selection result tuple.

    Returns:
        List of best model and statistics for each classifiers.

    """
    candidates, _ = ms_result
    candidates = [(i, c, cv['best_f1']) for i, c, cv in candidates]

    if mode == 'f1':
        f1s_mean = []
        for i, c, cv_best in candidates:
            # Iterate over splits to calculate average F1 score.
            f1s = [cv_best[f'split_{j}']['f1']
                   for j in range(int(len(cv_best)/2) - 1)]
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
    elif mode == 'balanced_accuracy':
        candidates, _ = ms_result
        # candidates = [(i, c, cv) for i, c, cv in candidates]
        balanced_accuracys_mean = []
        grid_results = []
        for i, c, cv in candidates:
            # parse every grid search result
            for key in cv:
                # Iterate over splits to calculate average F1 score for clf
                result = cv[key]
                balanced_accuracys = [
                    result[f'split_{j}']['balanced_accuracy'] for j in range(int(len(result)/2))]
                grid_results += [(i, c, result)]
                balanced_accuracys_mean += [
                    np.mean(np.nan_to_num(balanced_accuracys))]
        candidates = list(zip(grid_results, balanced_accuracys_mean))
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        best_candidate_per_clf = []
        for clf in CLASSIFIER_MODES:
            for (i, c, cv), balanced_accuracy_mean in candidates:
                if c[3] == clf:
                    if cv['param'] is not None:
                        cv['param'] = {k.split('__')[-1]: v
                                       for k, v in cv['param'].items()}

                    best_candidate_per_clf += [((i, c, cv),
                                                balanced_accuracy_mean)]
                    break
        return best_candidate_per_clf

        # raise NotImplementedError
    else:
        raise ValueError(f"Unknown mode: {mode}")


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
    '--use-smote/--no-use-smote',
    default=True)
@click.option(
    '--use-smote-first/--no-use-smote-first',
    default=False)
@click.option(
    '--all-best-confusion-matrices-test/--no-all-best-confusion-matrices-test',
    default=False)
@click.option(
    '--use-f1/--use-balanced-accuracy',
    default=True)
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
        use_smote,
        use_smote_first,
        all_best_confusion_matrices_test,
        use_f1,
        feature_kfold,
        random_state):
    """
    """
    if not os.path.exists(path_output_dir):
        os.mkdir(path_output_dir)

    model_selection_result = None
    with open(path_input_model_selection_result, 'rb') as f:
        model_selection_result = pickle.load(f)
    if use_f1:
        mode = "f1"
        best_candidate_per_clf = parse_model_selection_result(
            model_selection_result, mode)
    else:
        mode = "balanced_accuracy"
        best_candidate_per_clf = parse_model_selection_result(
            model_selection_result, mode)
    if all_best_confusion_matrices_test:
        pd.DataFrame(best_candidate_per_clf).to_csv(
            f"{path_output_dir}/best_clfs.csv")
        for (_, best_combination, best_cv_result), f1_mean in best_candidate_per_clf:
            best_scale_mode, best_impute_mode, best_outlier_mode, best_clf \
                = best_combination
            X_train, y_train = load_X_and_y(
                f"{path_input_preprocessed_data_dir}/"
                f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}_train.csv",
                col_y=feature_label)
            X_test, y_test = load_X_and_y(
                f"{path_input_preprocessed_data_dir}/"
                f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}_test.csv",
                col_y=feature_label)
            if use_smote:
                if use_smote_first:
                    clf = ClassifierHandler(
                        classifier_mode=best_clf,
                        params=best_cv_result['param'],
                        use_smote=False).clf
                    X_smote_train, y_smote_train = load_X_and_y(
                        f"{path_input_preprocessed_data_dir}/"
                        f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}_smote_train.csv",
                        col_y=feature_label)
                    splits = KFold_by_feature(
                        X=X_smote_train,
                        y=y_smote_train,
                        n_splits=5,
                        feature=feature_kfold,
                        random_state=random_state)
                    del best_cv_result['param']
                    plot_all_confusion_matrices(clf, X_smote_train, y_smote_train, X_test, y_test,
                                                path_output_dir, use_smote_first=True, use_rfe=False,
                                                use_f1=use_f1, splits=splits, classifier=best_clf)
                else:
                    clf = ClassifierHandler(
                        classifier_mode=best_clf,
                        params=best_cv_result['param'],
                        random_state=ModelSelectionConfig.RNG_SMOTE).clf
                    splits = KFold_by_feature(
                        X=X_train,
                        y=y_train,
                        n_splits=5,
                        feature=feature_kfold,
                        random_state=random_state)
                    del best_cv_result['param']
                    plot_all_confusion_matrices(clf, X_train, y_train, X_test, y_test,
                                                path_output_dir, use_smote_first=False,
                                                use_rfe=False, use_f1=use_f1, splits=splits,
                                                classifier=best_clf)
            else:
                clf = ClassifierHandler(
                    classifier_mode=best_clf,
                    params=best_cv_result['param'],
                    use_smote=False).clf
                splits = KFold_by_feature(
                    X=X_train,
                    y=y_train,
                    n_splits=5,
                    feature=feature_kfold,
                    random_state=random_state)
                del best_cv_result['param']
                plot_all_confusion_matrices(clf, X_train, y_train, X_test, y_test,
                                            path_output_dir, use_smote_first=False,
                                            use_rfe=False, use_f1=use_f1, splits=splits,
                                            classifier=best_clf)

        return

    best_candidate = max(best_candidate_per_clf, key=lambda x: x[1])
    _, best_combination, best_cv_result = best_candidate[0]
    best_scale_mode, best_impute_mode, best_outlier_mode, best_clf \
        = best_combination
    # print(best_combination)
    pd.DataFrame(best_candidate_per_clf).to_csv(
        f"{path_output_dir}/best_clfs.csv")

    # X_raw, _ = load_X_and_y(path_input_data_raw, col_y=feature_label)

    # X, y = load_X_and_y(
    #     f"{path_input_preprocessed_data_dir}/"
    #     f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}.csv",
    #     col_y=feature_label)
    # idxes_outlier = np.loadtxt(
    #     f"{path_input_preprocessed_data_dir}/"
    #     f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}"
    #     "_outlier_indices.txt",
    #     delimiter='\n',
    #     dtype=int)

    X_train, y_train = load_X_and_y(
        f"{path_input_preprocessed_data_dir}/"
        f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}_train.csv",
        col_y=feature_label)
    X_test, y_test = load_X_and_y(
        f"{path_input_preprocessed_data_dir}/"
        f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}_test.csv",
        col_y=feature_label)

    # visualize train data
    # Plot pairwise correlation heatmaps.
    plot_all_correlations(X_train=X_train, y_train=y_train, feature_label=feature_label,
                          path_output_dir=path_output_dir, use_smote_first=False, use_rfe=False)

    # Plot similarity matrix for the data points heatmap.
    plot_similarity_matrix(X_train=X_train, y_train=y_train,
                           path_output_dir=path_output_dir, use_smote_first=False, use_rfe=False)

    # Plot embedded data points.
    plot_all_embeddings(X_train=X_train, y_train=y_train, path_output_dir=path_output_dir,
                        random_state=random_state, use_smote_first=False, use_rfe=False)
    if use_smote:
        if use_smote_first:
            X_smote_train, y_smote_train = load_X_and_y(
                f"{path_input_preprocessed_data_dir}/"
                f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}_smote_train.csv",
                col_y=feature_label)

            splits = KFold_by_feature(
                X=X_smote_train,
                y=y_smote_train,
                n_splits=5,
                feature=feature_kfold,
                random_state=random_state)
            if feature_kfold is not None:
                X_smote_train = X_smote_train.drop([feature_kfold], axis=1)
                X_test = X_test.drop([feature_kfold], axis=1)

            clf = ClassifierHandler(
                classifier_mode=best_clf,
                params=best_cv_result['param'],
                use_smote=False).clf

            # data visualization for train data after smote
            # Plot pairwise correlation heatmaps.
            plot_all_correlations(X_train=X_smote_train, y_train=y_smote_train, feature_label=feature_label,
                                  path_output_dir=path_output_dir, use_smote_first=True, use_rfe=False)

            # Plot similarity matrix
            plot_similarity_matrix(X_train=X_smote_train, y_train=y_smote_train,
                                   path_output_dir=path_output_dir, use_smote_first=True, use_rfe=False)

            # Plot embedded data points.
            plot_all_embeddings(X_train=X_smote_train, y_train=y_smote_train, path_output_dir=path_output_dir,
                                random_state=random_state, use_smote_first=True, use_rfe=False)

            # RFE/SFS
            # Calculate and plot feature selection for the best model.
            if use_f1:
                mode = "f1"
            else:
                mode = "balanced_accuracy"
            sfs = get_selected_features(
                clf, X_smote_train, y_smote_train, mode, splits)
            plot_rfe_line(
                sfs,
                title="Recursive Feature Elimination",
                path_save=f"{path_output_dir}/rfe_smote.svg")
            plot_rfe_line_detailed(
                sfs,
                title="Recursive Feature Elimination",
                path_save=f"{path_output_dir}/rfe_smote_detailed.svg")
            pd.DataFrame(sfs.get_metric_dict()).transpose().reset_index().to_csv(
                f"{path_output_dir}/rfe_smote_result.csv", index=False)
            if len(sfs.k_feature_idx_) < 3:
                logging.warning(
                    f"Number of features from rfe is few: {sfs.k_feature_idx_}")
            else:
                logging.info(f"Features from rfe: {sfs.k_feature_idx_}")

            # use rfe results to plot curves, CM's, etc
            X_rfe_train = X_smote_train.iloc[:, list(sfs.k_feature_idx_)]
            X_rfe_test = X_test.iloc[:, list(sfs.k_feature_idx_)]
            plot_all_curves(clf, X_rfe_train, y_smote_train, X_rfe_test, y_test,
                            path_output_dir, use_smote_first=True, use_rfe=True, splits=splits)
            plot_all_confusion_matrices(clf, X_rfe_train, y_smote_train,
                                        X_rfe_test, y_test, path_output_dir,
                                        use_smote_first=True, use_rfe=True,
                                        use_f1=use_f1, splits=splits)
            # Plot embedded data points.
            plot_all_embeddings(X_train=X_rfe_train, y_train=y_smote_train, path_output_dir=path_output_dir,
                                random_state=random_state, use_smote_first=True, use_rfe=True)
            # Plot correlations for rfe
            plot_all_correlations(X_train=X_rfe_train, y_train=y_smote_train, feature_label=feature_label,
                                  path_output_dir=path_output_dir, use_smote_first=True, use_rfe=True)
            # Plot similarity matrix for rfe
            plot_similarity_matrix(X_train=X_rfe_train, y_train=y_smote_train,
                                   path_output_dir=path_output_dir, use_smote_first=True, use_rfe=True)

            # curves for best model no rfe
            plot_all_curves(clf, X_smote_train, y_smote_train, X_test, y_test,
                            path_output_dir, use_smote_first=True, use_rfe=False, splits=splits)

            # Plot confusion matrix with various metrics for validation.
            del best_cv_result['param']

            plot_all_confusion_matrices(clf, X_smote_train, y_smote_train,
                                        X_test, y_test, path_output_dir,
                                        use_smote_first=True, use_rfe=False,
                                        use_f1=use_f1, splits=splits)

        else:
            splits = KFold_by_feature(
                X=X_train,
                y=y_train,
                n_splits=5,
                feature=feature_kfold,
                random_state=random_state)
            if feature_kfold is not None:
                X_train = X_train.drop([feature_kfold], axis=1)
                X_test = X_test.drop([feature_kfold], axis=1)

            clf = ClassifierHandler(
                classifier_mode=best_clf,
                params=best_cv_result['param'],
                random_state=ModelSelectionConfig.RNG_SMOTE).clf

            # RFE/SFS
            # Calculate and plot feature selection for the best model.
            if use_f1:
                mode = "f1"
            else:
                mode = "balanced_accuracy"
            sfs = get_selected_features(clf, X_train, y_train, mode, splits)
            plot_rfe_line(
                sfs,
                title="Recursive Feature Elimination",
                path_save=f"{path_output_dir}/rfe_val.svg")
            plot_rfe_line_detailed(
                sfs,
                title="Recursive Feature Elimination",
                path_save=f"{path_output_dir}/rfe_val_detailed.svg")
            pd.DataFrame(sfs.get_metric_dict()).transpose().reset_index().to_csv(
                f"{path_output_dir}/rfe_result_val.csv", index=False)
            if len(sfs.k_feature_idx_) < 3:
                logging.warning(
                    f"Number of features from rfe is few: {sfs.k_feature_idx_}")
            else:
                logging.info(f"Features from rfe: {sfs.k_feature_idx_}")

            # use rfe results to plot curves, CM's, etc
            X_rfe_train = X_train.iloc[:, list(sfs.k_feature_idx_)]
            X_rfe_test = X_test.iloc[:, list(sfs.k_feature_idx_)]
            plot_all_curves(clf, X_rfe_train, y_train, X_rfe_test, y_test,
                            path_output_dir, use_smote_first=False, use_rfe=True, splits=splits)
            plot_all_confusion_matrices(clf, X_rfe_train, y_train,
                                        X_rfe_test, y_test, path_output_dir,
                                        use_smote_first=False, use_rfe=True,
                                        use_f1=use_f1, splits=splits)
            # Plot embedded data points.
            plot_all_embeddings(X_train=X_rfe_train, y_train=y_train, path_output_dir=path_output_dir,
                                random_state=random_state, use_smote_first=False, use_rfe=True)
            # Plot correlations for rfe
            plot_all_correlations(X_train=X_rfe_train, y_train=y_train, feature_label=feature_label,
                                  path_output_dir=path_output_dir, use_smote_first=False, use_rfe=True)
            # Plot similarity matrix for rfe
            plot_similarity_matrix(X_train=X_rfe_train, y_train=y_train,
                                   path_output_dir=path_output_dir, use_smote_first=False, use_rfe=True)

            # curves for best model no rfe
            plot_all_curves(clf, X_train, y_train, X_test, y_test,
                            path_output_dir, use_smote_first=False, use_rfe=False, splits=splits)

            # Plot confusion matrix with various metrics for validation.
            del best_cv_result['param']
            plot_all_confusion_matrices(clf, X_train, y_train,
                                        X_test, y_test, path_output_dir,
                                        use_smote_first=False, use_rfe=False,
                                        use_f1=use_f1, splits=splits)
    else:
        splits = KFold_by_feature(
            X=X_train,
            y=y_train,
            n_splits=5,
            feature=feature_kfold,
            random_state=random_state)
        if feature_kfold is not None:
            X_train = X_train.drop([feature_kfold], axis=1)
            X_test = X_test.drop([feature_kfold], axis=1)

        clf = ClassifierHandler(
            classifier_mode=best_clf,
            params=best_cv_result['param'],
            use_smote=False).clf

        # RFE/SFS
        # Calculate and plot feature selection for the best model.
        if use_f1:
            mode = "f1"
        else:
            mode = "balanced_accuracy"
        sfs = get_selected_features(clf, X_train, y_train, mode, splits)
        plot_rfe_line(
            sfs,
            title="Recursive Feature Elimination",
            path_save=f"{path_output_dir}/rfe_val.svg")
        plot_rfe_line_detailed(
            sfs,
            title="Recursive Feature Elimination",
            path_save=f"{path_output_dir}/rfe_val_detailed.svg")
        pd.DataFrame(sfs.get_metric_dict()).transpose().reset_index().to_csv(
            f"{path_output_dir}/rfe_result_val.csv", index=False)
        if len(sfs.k_feature_idx_) < 3:
            logging.warning(
                f"Number of features from rfe is few: {sfs.k_feature_idx_}")
        else:
            logging.info(f"Features from rfe: {sfs.k_feature_idx_}")

        # use rfe results to plot curves, CM's, etc
        X_rfe_train = X_train.iloc[:, list(sfs.k_feature_idx_)]
        X_rfe_test = X_test.iloc[:, list(sfs.k_feature_idx_)]
        plot_all_curves(clf, X_rfe_train, y_train, X_rfe_test, y_test,
                        path_output_dir, use_smote_first=False, use_rfe=True, splits=splits)
        plot_all_confusion_matrices(clf, X_rfe_train, y_train,
                                    X_rfe_test, y_test, path_output_dir,
                                    use_smote_first=False, use_rfe=True,
                                    use_f1=use_f1, splits=splits)
        # Plot embedded data points.
        plot_all_embeddings(X_train=X_rfe_train, y_train=y_train, path_output_dir=path_output_dir,
                            random_state=random_state, use_smote_first=False, use_rfe=True)
        # Plot correlations for rfe
        plot_all_correlations(X_train=X_rfe_train, y_train=y_train, feature_label=feature_label,
                              path_output_dir=path_output_dir, use_smote_first=False, use_rfe=True)
        # Plot similarity matrix for rfe
        plot_similarity_matrix(X_train=X_rfe_train, y_train=y_train,
                               path_output_dir=path_output_dir, use_smote_first=False, use_rfe=True)

        # curves for best model no rfe
        plot_all_curves(clf, X_train, y_train, X_test, y_test,
                        path_output_dir, use_smote_first=False, use_rfe=False, splits=splits)

        # Plot confusion matrix with various metrics for validation.
        del best_cv_result['param']
        plot_all_confusion_matrices(clf, X_train, y_train,
                                    X_test, y_test, path_output_dir,
                                    use_smote_first=False, use_rfe=False,
                                    use_f1=use_f1, splits=splits)


if __name__ == '__main__':
    main()
