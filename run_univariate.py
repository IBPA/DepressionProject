# -*- coding: utf-8 -*-
"""Run univariate feature selection and
    compare to rfe output

Authors:
    Fangzhou Li - fzli@ucdavis.edu
    Arielle Yoo - asmyoo@ucdavis.edu

python -u -m DepressionProjectNew.run_univariate \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/results.pkl \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental/preprocessed \
    ./DepressionProjectNew/output/10MVIout/data_cleaned_encoded_18_yesmental.csv \
    ./DepressionProjectNew/output/10MVIout/output_18_yesmental \
    y18CH_Dep_YN_216m

"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
import click
from sklearn.metrics import precision_score
from ast import literal_eval

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
    plot_embedded_scatter,
    plot_rfe_line,
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
                   for j in range(int(len(cv_best)/2))]
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
    best_candidate = max(best_candidate_per_clf, key=lambda x: x[1])
    _, best_combination, best_cv_result = best_candidate[0]
    best_scale_mode, best_impute_mode, best_outlier_mode, best_clf \
        = best_combination
    # pd.DataFrame(best_candidate_per_clf).to_csv(
    #    f"{path_output_dir}/best_clfs.csv")

    # X_raw, _ = load_X_and_y(path_input_data_raw, col_y=feature_label)

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
        random_state=ModelSelectionConfig.RNG_SMOTE).clf

    # Calculate and plot feature selection for the best model.
    # sfs = get_selected_features(clf, X, y, splits)
    # plot_rfe_line(
    #     sfs,
    #     title="Recursive Feature Elimination",
    #     path_save=f"{path_output_dir}/rfe.png")
    # pd.DataFrame(sfs.get_metric_dict()).transpose().reset_index().to_csv(
    #     f"{path_output_dir}/rfe_result.csv", index=False)

    # perform univariate feature selection
    fts_scores = get_univariate_features_all(X, y)
    # get only fts
    fts_univariate = [item[0] for item in fts_scores]
    # logging.info(fts_univariate)

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
    # logging.info(rfe_fts_ordered)
    df_fs = pd.DataFrame(list(zip(fts_univariate, rfe_fts_ordered)),
                         columns=['Univariate', 'RFE'])
    df_fs.to_csv(f"{path_output_dir}/feature_selection_ordered.csv",
                 index=False)

    # print what top 10 match
    top10match = [x for x in fts_univariate[:10] if x in rfe_fts_ordered[:10]]
    logging.info("Top 10 matching: "
                 f"{top10match}")

    # organize into table
    # row is column, column rank in RFE, column rank in univariate
    # should just be 0 to end
    indices_rfe = [rfe_fts_ordered.index(x) for x in rfe_fts_ordered]
    indices_univariate = [fts_univariate.index(x) for x in rfe_fts_ordered]
    # print(indices_rfe)
    # print(indices_univariate)
    df = pd.DataFrame(list(zip(rfe_fts_ordered, indices_rfe, indices_univariate)),
                      columns=['Variable', 'RFE Index', 'Univariate Index'])
    df.to_csv(f"{path_output_dir}/feature_selection_indices.csv",
              index=False)


if __name__ == '__main__':
    main()
