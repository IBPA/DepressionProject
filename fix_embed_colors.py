# -*- coding: utf-8 -*-
"""Run univariate feature selection and
    compare to rfe output

Authors:
    Fangzhou Li - fzli@ucdavis.edu
    Arielle Yoo - asmyoo@ucdavis.edu

python -u -m DepressionProjectNew.fix_embed_colors \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/results.pkl \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/preprocessed \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/data_cleaned_encoded.csv \
    ./DepressionProjectNew/output/pval_filter_60_MVI/output_12_yesmental/ \
    y12CH_Dep_YN_144m

"""
import os
import pickle
import logging
from statistics import stdev

import numpy as np
import pandas as pd
import click
import scipy as sp
from sklearn.metrics import precision_score
from ast import literal_eval
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from kneed import KneeLocator
from kneebow.rotor import Rotor
import matplotlib.pyplot as plt

from msap.modeling.model_evaluation.statistics import (
    get_embedded_data,
    get_selected_features,
    get_univariate_features_all,
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
    plot_rfe_line,
    plot_rfe_line_from_dataframe,
    plot_curves,
    plot_confusion_matrix)
from msap.modeling.configs import (
    ModelSelectionConfig)

from .plot_rfe_fang import get_parsimonious

from .run_analysis import (
    parse_model_selection_result,
    plot_all_confusion_matrices,
    plot_all_curves,
    plot_all_correlations,
    plot_similarity_matrix)

from .run_univariate import (
    get_knee,
    get_kneebow,

)

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
DEFAULT_VARIABLE_INFO = './DepressionProjectNew/data/Variables052122.csv'
# DEFAULT_PREPROCESSED = './output/preprocessed_data_without_temporal_12to18ave.csv'
FONT_SIZE_LABEL = 30


# added hue order to MSAP code for plotting


def plot_embedded_scatter(
        X: pd.DataFrame,
        y: pd.Series,
        title: str = None,
        path_save: str = None):
    fig, ax = plt.subplots(figsize=(15, 15))

    g = sns.scatterplot(
        data=pd.concat([X, y], axis=1),
        x=X.columns[0],
        y=X.columns[1],
        hue=y.name,
        hue_order=['Not Depressed', 'Depressed'],
        legend="full",
        s=50)
    ax.set_xlabel(
        xlabel=ax.get_xlabel(),
        fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(
        ylabel=ax.get_ylabel(),
        fontsize=FONT_SIZE_LABEL)
    plt.setp(g.get_legend().get_texts(), fontsize='20')
    plt.setp(g.get_legend().get_title(), fontsize='20')

    if title is not None:
        ax.set_title(title, fontsize=40, fontweight='bold')

    if path_save is not None:
        fig.savefig(path_save)
        plt.close()
    else:
        plt.show()


# same as code from run_analysis.py


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
        use_f1,
        feature_kfold,
        random_state):
    """
    """
    if not os.path.exists(path_output_dir):
        os.mkdir(path_output_dir)

    # replot embeddings in new folder
    embed_folder = f"{path_output_dir}/embed_fixed"
    if not os.path.exists(embed_folder):
        os.makedirs(embed_folder)

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
    best_candidate = max(best_candidate_per_clf, key=lambda x: x[1])
    _, best_combination, best_cv_result = best_candidate[0]
    best_scale_mode, best_impute_mode, best_outlier_mode, best_clf \
        = best_combination

    X_train, y_train = load_X_and_y(
        f"{path_input_preprocessed_data_dir}/"
        f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}_train.csv",
        col_y=feature_label)

    # # swap depression with non depression in first element
    # tmp_x = X_train.iloc[0].copy()
    # tmp_y = y_train.iloc[0].copy()
    # # get next location where non depression is
    # idx_non_depression = y_train[y_train == 0].index[0]
    # # swap depression with non depression in first element
    # X_train.iloc[0] = X_train.iloc[idx_non_depression]
    # y_train.iloc[0] = y_train.iloc[idx_non_depression]
    # X_train.iloc[idx_non_depression] = tmp_x
    # y_train.iloc[idx_non_depression] = tmp_y

    plot_all_embeddings(X_train=X_train, y_train=y_train, path_output_dir=embed_folder,
                        random_state=random_state, use_smote_first=False, use_rfe=False)

    X_test, y_test = load_X_and_y(
        f"{path_input_preprocessed_data_dir}/"
        f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}_test.csv",
        col_y=feature_label)

    if use_smote:
        if use_smote_first:
            X_smote_train, y_smote_train = load_X_and_y(
                f"{path_input_preprocessed_data_dir}/"
                f"{best_scale_mode}_{best_impute_mode}_{best_outlier_mode}_smote_train.csv",
                col_y=feature_label)

            # # swap depression with non depression in first element
            # tmp_x = X_smote_train.iloc[0].copy()
            # tmp_y = y_smote_train.iloc[0].copy()
            # # get next location where non depression is
            # idx_non_depression = y_smote_train[y_smote_train == 0].index[0]
            # # swap depression with non depression in first element
            # X_smote_train.iloc[0] = X_smote_train.iloc[idx_non_depression]
            # y_smote_train.iloc[0] = y_smote_train.iloc[idx_non_depression]
            # X_smote_train.iloc[idx_non_depression] = tmp_x
            # y_smote_train.iloc[idx_non_depression] = tmp_y

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

            # Plot embedded data points.
            plot_all_embeddings(X_train=X_smote_train, y_train=y_smote_train, path_output_dir=embed_folder,
                                random_state=random_state, use_smote_first=True, use_rfe=False)

            # reset X_train and y_train
            X_train = X_smote_train
            y_train = y_smote_train
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

    rfe = pd.read_csv(f"{path_output_dir}/rfe_result_val.csv")
    # logging.info(rfe)
    # reverse list of rfe features
    rfe_fts = rfe["feature_names"][::-1].reset_index(drop=True)
    # read as list
    rfe_fts = rfe_fts.apply(lambda x: literal_eval(str(x)))
    # get into same format as fts_scores
    rfe_fts_ordered = []  # most important is first
    rfe_fts_ordered += list(rfe_fts[0])
    # loop through results and grab next unique value
    for i in range(len(rfe_fts) - 1):
        rfe_fts_ordered += [x for x in rfe_fts[i+1] if x not in rfe_fts[i]]

    # get feature importance for rfe results (from fang's code)
    rfe['feature_idx'] = rfe['feature_idx'].apply(literal_eval)
    rfe['cv_scores'] = rfe['cv_scores'].apply(
        lambda x: np.fromstring(x[1:-1], dtype=float, sep=' ')
    )
    rfe['feature_names'] = rfe['feature_names'].apply(
        literal_eval
    )
    k = get_parsimonious(rfe)[0]
    selected_fts = rfe_fts_ordered[:k]
    logging.info(f"Selected features: {selected_fts}")
    X_train_rfe = X_train[selected_fts]
    X_test_rfe = X_test[selected_fts]
    # Plot embedded data points.
    if use_smote:
        if use_smote_first:
            plot_all_embeddings(X_train=X_train_rfe, y_train=y_train, path_output_dir=embed_folder,
                                random_state=random_state, use_smote_first=True, use_rfe=True)
        else:
            plot_all_embeddings(X_train=X_train_rfe, y_train=y_train, path_output_dir=embed_folder,
                                random_state=random_state, use_smote_first=False, use_rfe=True)
    else:
        plot_all_embeddings(X_train=X_train_rfe, y_train=y_train, path_output_dir=embed_folder,
                            random_state=random_state, use_smote_first=False, use_rfe=True)

    # replot rfe with selected features from knee method
    knee_folder = f"{embed_folder}/kneebow"
    if not os.path.exists(knee_folder):
        os.makedirs(knee_folder)
    # k = get_knee(rfe, f"{knee_folder}/knee_info.svg")[0]
    k = get_kneebow(rfe, f"{knee_folder}/knee_info.svg")
    selected_fts = rfe_fts_ordered[:k]
    logging.info(f"Selected features knee: {selected_fts}")
    X_train_knee = X_train[selected_fts]
    X_test_knee = X_test[selected_fts]
    splits = KFold_by_feature(
        X_train_knee, y_train, n_splits=5, feature=feature_kfold, random_state=random_state)
    if feature_kfold is not None:
        X_train_knee = X_train_knee.drop([feature_kfold], axis=1)
        X_test = X_test.drop([feature_kfold], axis=1)
        X_test_knee = X_test_knee.drop([feature_kfold], axis=1)
    if use_smote:
        if use_smote_first:
            if use_f1:
                mode = "f1"
            else:
                mode = "balanced_accuracy"
            # Plot embedded data points.
            plot_all_embeddings(X_train=X_train_knee, y_train=y_train, path_output_dir=knee_folder,
                                random_state=random_state, use_smote_first=True, use_rfe=True)

        else:
            if use_f1:
                mode = "f1"
            else:
                mode = "balanced_accuracy"
            # Plot embedded data points.
            plot_all_embeddings(X_train=X_train_knee, y_train=y_train, path_output_dir=knee_folder,
                                random_state=random_state, use_smote_first=False, use_rfe=True)

    else:
        if use_f1:
            mode = "f1"
        else:
            mode = "balanced_accuracy"
        # Plot embedded data points.
        plot_all_embeddings(X_train=X_train_knee, y_train=y_train, path_output_dir=knee_folder,
                            random_state=random_state, use_smote_first=False, use_rfe=True)


if __name__ == '__main__':
    main()
