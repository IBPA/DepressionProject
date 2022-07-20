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
    plot_embedded_scatter,
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
    plot_all_embeddings,
    plot_all_curves,
    plot_all_correlations,
    plot_similarity_matrix)

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


def make_readable(want_readable, variable_filepath=DEFAULT_VARIABLE_INFO):
    """Make variable names readable.

    Args:
        want_readable: List of variable names to make readable.
        variable_filepath: String of filepath of variable info

    Returns:
        result: List of readable variable names.

    """
    df_variable_info = pd.read_csv(
        variable_filepath, dtype='str', encoding='unicode_escape')
    relabeled = []
    variable_description = []
    unfound = []

    # make variable description
    descriptions = ["{} ({})".format(a_, b_) for a_, b_ in zip(list(
        df_variable_info['Variable Label'].str.strip()), list(df_variable_info['Coding_details'].str.strip()))]

    mapper = dict(zip(
        list(df_variable_info['RelabeledName'].str.strip()),
        descriptions
    ))

    for name in want_readable:
        if len(name.split("_")) > 2:
            label_in_variable_info = "_".join(name.split("_")[0:2])
            var_map = map(mapper.get, [label_in_variable_info])
            var_desc = list(var_map)[0]
            if var_desc == None:
                # print(
                #     f"Len of name is 3 but not found in mapper if sliced: {name}")
                unfound.append(name)
                continue
            relabeled.append(name)
            variable_description.append(" ".join([var_desc, name]))
        else:
            label_in_variable_info = name
            var_map = map(mapper.get, [label_in_variable_info])
            var_desc = list(var_map)[0]
            # if var_desc == None:
            #     print(
            #         f"Len of name is <=2 but not found in mapper if sliced: {name}")
            #     unfound.append(name)
            #     continue
            relabeled.append(name)
            variable_description.append(" ".join([var_desc, name]))

    for name in unfound:  # assume these are in info somewhere
        # print(f"Looking for: {name}")
        relabeled.append(name)
        label_in_variable_info = name
        var_map = map(mapper.get, [label_in_variable_info])
        var_desc = list(var_map)[0]
        variable_description.append(" ".join([var_desc, name]))

    mapper = dict(zip(
        relabeled,
        variable_description))

    result = [mapper.get(item, item) for item in want_readable]
    return result


def has_feature_importance(clf):
    """Check if classifier has feature importance.

    Args:
        clf: Classifier object.

    Returns:
        Boolean.

    """
    return hasattr(clf, 'feature_importances_')


def get_feature_importance(clf):
    """Get feature importance.

    Args:
        clf: Classifier object.

    Returns:
        importances: Dataframe of feature names and importances

    """
    names = list(clf.feature_names_in_)
    imp = list(clf.feature_importances_)
    data = list(zip(names, imp))
    return pd.DataFrame(data=data, columns=["Variable", "Feature Importance"])


def get_knee(rfe_result, filepath=None):
    """Get knee of RFE.

    Args:
        rfe_result: Result of RFE.

    Returns:
        knee: Integer index of knee.

    """
    n_features = len(rfe_result.loc[0, 'feature_idx'])
    # reverse order so indices are increasing
    x = rfe_result['index'][::-1].tolist()
    y = rfe_result['avg_score'][::-1].tolist()
    # x = rfe_result['index'].tolist()
    # y = rfe_result['avg_score'].tolist()
    # print(y)
    kl = KneeLocator(x, y, curve='concave',
                     direction='increasing', S=15)
    # kl = KneeLocator(x, y, curve='concave',
    #                  direction='increasing',
    #                  interp_method="polynomial", polynomial_degree=4)
    # high sensitivity
    # kl = KneeLocator(x, y, curve='concave',
    #                  direction='increasing', S=50)
    print(kl.knee)
    print(kl.all_knees)
    if filepath is not None:
        plt.tight_layout()
        # code from kneed plot knee function
        plt.title("Normalized Knee Point")
        plt.plot(kl.x_normalized, kl.y_normalized,
                 "b", label="normalized curve")
        plt.plot(kl.x_difference, kl.y_difference,
                 "r", label="difference curve")
        plt.plot(kl.x_difference_maxima, kl.Tmx, "g", label="threshold")
        plt.xticks(
            np.arange(kl.x_normalized.min(), kl.x_normalized.max() + 0.1, 0.1)
        )
        plt.yticks(
            np.arange(kl.y_difference.min(), kl.y_normalized.max() + 0.1, 0.1)
        )

        plt.vlines(
            kl.norm_knee,
            plt.ylim()[0],
            plt.ylim()[1],
            linestyles="--",
            label=f"knee/elbow {kl.knee}",
        )
        plt.legend(loc="best")
        plt.savefig(filepath)
        plt.close()

    return kl.knee, kl.knee_y


def get_kneebow(rfe_result, filepath=None, test=False):
    """Get knee of RFE.

    Args:
        rfe_result: Result of RFE.
        test: Boolean. If true, use test data

    Returns:
        knee: Integer index of knee.

    """
    # reverse order so indices are increasing
    x = rfe_result['index'][::-1].tolist()
    y = rfe_result['avg_score'][::-1].tolist()
    # x = rfe_result['index'].tolist()
    # y = rfe_result['avg_score'].tolist()
    data = [list(z) for z in zip(x, y)]
    rotor = Rotor()
    rotor.fit_rotate(data)
    index = rotor.get_knee_index()
    print(index)
    # get elbow for test
    rfe_index = x[index]
    print(rfe_index)
    if test:
        # test from kneebow package
        data = np.array([
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [6, 6],
            [7, 7],
            [8, 8],
            [9, 16],
            [10, 32],
            [11, 64],
            [12, 128],
            [13, 256],
            [14, 512]
        ])
        rotor = Rotor()
        rotor.fit_rotate(data)
        index = rotor.get_elbow_index()
        rfe_index = data[index, 0]
    if filepath is not None:
        plt.tight_layout()
        # code from kneebow plot knee function
        plt.title("Normalized Knee Point")
        plt.plot(rotor._data[:, 0], rotor._data[:, 1],
                 c='b', label="normalized curve")
        plt.vlines(rotor._data[index, 0], ymin=rotor._data[:, 1].min(),
                   ymax=rotor._data[:, 1].max(), colors='r', linestyles="--",
                   label=f"knee/elbow {rfe_index}")
        plt.legend(loc="best")
        plt.savefig(filepath)
        plt.close()
    return rfe_index


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

    clf.fit(X_train, y_train)
    if has_feature_importance(clf.named_steps[best_clf]):
        feature_importance = get_feature_importance(clf.named_steps[best_clf])
        feature_importance.to_csv(
            f"{path_output_dir}/feature_importance.csv",
            index=False)
        feature_importance_readable = pd.DataFrame(list(zip(make_readable(
            feature_importance["Variable"]), feature_importance["Feature Importance"])), columns=feature_importance.columns)
        feature_importance_readable.to_csv(
            f"{path_output_dir}/feature_importance_readable.csv",
            index=False)

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

    # splits = KFold_by_feature(
    #     X=X,
    #     y=y,
    #     n_splits=5,
    #     feature=feature_kfold,
    #     random_state=random_state)
    if feature_kfold is not None:
        X = X.drop([feature_kfold], axis=1)

    # clf = ClassifierHandler(
    #     classifier_mode=best_clf,
    #     params=best_cv_result['param'],
    #     random_state=ModelSelectionConfig.RNG_SMOTE).clf

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
    df_fs_readable = pd.DataFrame(list(zip(make_readable(fts_univariate), make_readable(rfe_fts_ordered))),
                                  columns=['Univariate', 'RFE'])
    df_fs_readable.to_csv(f"{path_output_dir}/feature_selection_ordered_readable.csv",
                          index=False)

    # print what top 10 match
    top10match = [x for x in fts_univariate[:10] if x in rfe_fts_ordered[:10]]
    logging.info("Top 10 matching: "
                 f"{top10match}")

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
    clf.fit(X_train_rfe, y_train)
    if has_feature_importance(clf.named_steps[best_clf]):
        feature_importance = get_feature_importance(clf.named_steps[best_clf])
        feature_importance.to_csv(
            f"{path_output_dir}/feature_importance_rfe.csv",
            index=False)
        feature_importance_readable = pd.DataFrame(list(zip(make_readable(
            feature_importance["Variable"]), feature_importance["Feature Importance"])), columns=feature_importance.columns)
        feature_importance_readable.to_csv(
            f"{path_output_dir}/feature_importance_readable_rfe.csv",
            index=False)

    # replot rfe with selected features from knee method
    knee_folder = f"{path_output_dir}/knee"
    if not os.path.exists(knee_folder):
        os.makedirs(knee_folder)
    # k = get_knee(rfe, f"{knee_folder}/knee_info.svg")[0]
    k = get_kneebow(rfe, f"{knee_folder}/knee_info.svg")
    selected_fts = rfe_fts_ordered[:k]
    logging.info(f"Selected features knee: {selected_fts}")
    X_train_knee = X_train[selected_fts]
    X_test_knee = X_test[selected_fts]
    clf.fit(X_train_knee, y_train)
    if has_feature_importance(clf.named_steps[best_clf]):
        feature_importance = get_feature_importance(clf.named_steps[best_clf])
        feature_importance.to_csv(
            f"{path_output_dir}/feature_importance_knee.csv",
            index=False)
        feature_importance_readable = pd.DataFrame(list(zip(make_readable(
            feature_importance["Variable"]), feature_importance["Feature Importance"])), columns=feature_importance.columns)
        feature_importance_readable.to_csv(
            f"{path_output_dir}/feature_importance_readable_knee.csv",
            index=False)
    plot_rfe_line_from_dataframe(
        rfe, k, title='Sequential Feature Selection', path_save=f"{knee_folder}/rfe_val_detailed.svg")
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
            plot_all_curves(clf, X_train_knee, y_train, X_test_knee, y_test,
                            knee_folder, use_smote_first=True, use_rfe=True, splits=splits)
            plot_all_confusion_matrices(clf, X_train_knee, y_train,
                                        X_test_knee, y_test, knee_folder,
                                        use_smote_first=True, use_rfe=True,
                                        use_f1=use_f1, splits=splits)
            # Plot embedded data points.
            plot_all_embeddings(X_train=X_train_knee, y_train=y_train, path_output_dir=knee_folder,
                                random_state=random_state, use_smote_first=True, use_rfe=True)

        else:
            if use_f1:
                mode = "f1"
            else:
                mode = "balanced_accuracy"
            plot_all_curves(clf, X_train_knee, y_train, X_test_knee, y_test,
                            knee_folder, use_smote_first=False, use_rfe=True, splits=splits)
            plot_all_confusion_matrices(clf, X_train_knee, y_train,
                                        X_test_knee, y_test, knee_folder,
                                        use_smote_first=False, use_rfe=True,
                                        use_f1=use_f1, splits=splits)
            # Plot embedded data points.
            plot_all_embeddings(X_train=X_train_knee, y_train=y_train, path_output_dir=knee_folder,
                                random_state=random_state, use_smote_first=False, use_rfe=True)

    else:
        if use_f1:
            mode = "f1"
        else:
            mode = "balanced_accuracy"
        plot_all_curves(clf, X_train_knee, y_train, X_test_knee, y_test,
                        knee_folder, use_smote_first=False, use_rfe=True, splits=splits)
        plot_all_confusion_matrices(clf, X_train_knee, y_train,
                                    X_test_knee, y_test, knee_folder,
                                    use_smote_first=False, use_rfe=True,
                                    use_f1=use_f1, splits=splits)
        # Plot embedded data points.
        plot_all_embeddings(X_train=X_train_knee, y_train=y_train, path_output_dir=knee_folder,
                            random_state=random_state, use_smote_first=False, use_rfe=True)

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
    df_fs_readable = pd.DataFrame(list(zip(make_readable(rfe_fts_ordered), indices_rfe, indices_univariate)),
                                  columns=['Variable', 'RFE Index', 'Univariate Index'])
    df_fs_readable.to_csv(f"{path_output_dir}/feature_selection_indices_readable.csv",
                          index=False)

    # plot with pearson and spearman info?
    # assumes pearson and spearman info is in output folder
    pearson_corr = pd.read_csv(
        f"{path_output_dir}/pc_rank_pearson_train.csv")
    # print(pearson_corr)
    pearson_corr = pearson_corr.rename(columns={
                                       "Unnamed: 0": "Variable", "corr": "pearson_corr", "p-value": "pearson_pvalue"})
    spearman_corr = pd.read_csv(
        f"{path_output_dir}/pc_rank_spearman_train.csv")
    spearman_corr = spearman_corr.rename(columns={
                                         "Unnamed: 0": "Variable", "corr": "spearman_corr", "p-value": "spearman_pvalue"})

    # plot Pearson and Spearman pdfs
    # Pearson
    sns.set(style='whitegrid', font_scale=1.5)
    ax = sns.kdeplot(pearson_corr['pearson_corr'],
                     label='Pearson Correlation', shade=True)
    # ax.legend(loc='upper right')
    ax.figure.tight_layout()
    plt.axvline(pearson_corr['pearson_corr'].mean(), color='r', linestyle='--')
    plt.axvline(pearson_corr['pearson_corr'].mean() -
                stdev(pearson_corr['pearson_corr']), color='b', linestyle='--')
    plt.axvline(pearson_corr['pearson_corr'].mean() +
                stdev(pearson_corr['pearson_corr']), color='b', linestyle='--')
    ax.figure.savefig(f"{path_output_dir}/pearson.svg", bbox_inches='tight')
    plt.close()
    # Spearman
    ax = sns.kdeplot(spearman_corr['spearman_corr'],
                     label='Spearman Correlation', shade=True)
    # ax.legend(loc='upper right')
    ax.figure.tight_layout()
    plt.axvline(spearman_corr['spearman_corr'].mean(),
                color='r', linestyle='--')
    plt.axvline(spearman_corr['spearman_corr'].mean() -
                stdev(spearman_corr['spearman_corr']), color='b', linestyle='--')
    plt.axvline(spearman_corr['spearman_corr'].mean() + stdev(spearman_corr['spearman_corr']),
                color='b', linestyle='--')
    ax.figure.savefig(f"{path_output_dir}/spearman.svg", bbox_inches='tight')
    plt.close()

    # plot absolute value of Pearson and Spearman pdfs
    # Pearson
    sns.set(style='whitegrid', font_scale=1.5)
    ax = sns.kdeplot(abs(pearson_corr['pearson_corr']),
                     label='Pearson Correlation', shade=True)
    # ax.legend(loc='upper right')
    ax.figure.tight_layout()
    plt.axvline(abs(pearson_corr['pearson_corr']
                    ).mean(), color='r', linestyle='--')
    plt.axvline(abs(pearson_corr['pearson_corr']).mean() -
                stdev(abs(pearson_corr['pearson_corr'])), color='b',
                linestyle='--')
    plt.axvline(abs(pearson_corr['pearson_corr']).mean() +
                stdev(abs(pearson_corr['pearson_corr'])),
                color='b', linestyle='--')
    ax.figure.savefig(f"{path_output_dir}/pearson_abs.svg",
                      bbox_inches='tight')
    plt.close()
    # Spearman
    ax = sns.kdeplot(abs(spearman_corr['spearman_corr']),
                     label='Spearman Correlation', shade=True)
    # ax.legend(loc='upper right')
    ax.figure.tight_layout()
    plt.axvline(abs(spearman_corr['spearman_corr']
                    ).mean(), color='r', linestyle='--')
    plt.axvline(abs(spearman_corr['spearman_corr']).mean() -
                stdev(abs(spearman_corr['spearman_corr'])), color='b', linestyle='--')
    plt.axvline(abs(spearman_corr['spearman_corr']).mean() +
                stdev(abs(spearman_corr['spearman_corr'])),
                color='b', linestyle='--')
    ax.figure.savefig(
        f"{path_output_dir}/spearman_abs.svg", bbox_inches='tight')
    plt.close()

    # print(spearman_corr)
    dataframes = [df, pearson_corr, spearman_corr]
    df_corr = reduce(lambda left, right: pd.merge(left, right, on=['Variable'],
                                                  how='outer'), dataframes)
    # print(df_corr)
    df_corr.to_csv(f"{path_output_dir}/feature_selection_corr.csv",
                   index=False)
    df_corr_readable = df_corr.copy()
    df_corr_readable['Variable'] = make_readable(df_corr_readable['Variable'])
    df_corr_readable.to_csv(f"{path_output_dir}/feature_selection_corr_readable.csv",
                            index=False)

    # plot ranks from univariate vs rfe
    ax = sns.scatterplot(data=df_corr, x='RFE Index', y='Univariate Index')
    ax.figure.tight_layout()
    ax.figure.savefig(f"{path_output_dir}/rfe_vs_univariate.svg",
                      bbox_inches='tight')
    plt.close()

    # plot pearson correlation and rfe
    df_corr['pearson_corr_abs'] = abs(df_corr['pearson_corr'])
    ax = sns.scatterplot(data=df_corr, x='pearson_corr_abs', y='RFE Index')
    ax.figure.tight_layout()
    ax.figure.savefig(f"{path_output_dir}/pearson_vs_rfe.svg",
                      bbox_inches='tight')
    plt.close()

    ax = sns.scatterplot(data=df_corr, x='RFE Index', y='pearson_corr_abs')
    ax.figure.tight_layout()
    ax.figure.savefig(f"{path_output_dir}/rfe_vs_pearson.svg",
                      bbox_inches='tight')
    plt.close()

    # plot spearman correlation and rfe
    df_corr['spearman_corr_abs'] = abs(df_corr['spearman_corr'])
    ax = sns.scatterplot(data=df_corr, x='spearman_corr_abs', y='RFE Index')
    ax.figure.tight_layout()
    ax.figure.savefig(f"{path_output_dir}/spearman_vs_rfe.svg",
                      bbox_inches='tight')
    plt.close()

    ax = sns.scatterplot(data=df_corr, x='RFE Index', y='spearman_corr_abs')
    ax.figure.tight_layout()
    ax.figure.savefig(f"{path_output_dir}/rfe_vs_spearman.svg",
                      bbox_inches='tight')
    plt.close()

    # plot pearson correlation and univariate
    ax = sns.scatterplot(
        data=df_corr, x='pearson_corr_abs', y='Univariate Index')
    ax.figure.tight_layout()
    ax.figure.savefig(f"{path_output_dir}/pearson_vs_univariate.svg",
                      bbox_inches='tight')
    plt.close()

    ax = sns.scatterplot(
        data=df_corr, x='Univariate Index', y='pearson_corr_abs')
    ax.figure.tight_layout()
    ax.figure.savefig(f"{path_output_dir}/univariate_vs_pearson.svg",
                      bbox_inches='tight')
    plt.close()

    # plot spearman correlation and univariate
    ax = sns.scatterplot(
        data=df_corr, x='spearman_corr_abs', y='Univariate Index')
    ax.figure.tight_layout()
    ax.figure.savefig(f"{path_output_dir}/spearman_vs_univariate.svg",
                      bbox_inches='tight')
    plt.close()

    ax = sns.scatterplot(
        data=df_corr, x='Univariate Index', y='spearman_corr_abs')
    ax.figure.tight_layout()
    ax.figure.savefig(f"{path_output_dir}/univariate_vs_spearman.svg",
                      bbox_inches='tight')
    plt.close()

    # calc kendall tau
    tau, pval = kendalltau(df_corr['RFE Index'], df_corr['Univariate Index'])
    # print(df_corr['RFE Index'].dtype)
    # print(df_corr['Univariate Index'].dtype)
    print(f"Tau: {tau}")
    print(f"P-value: {pval}")


if __name__ == '__main__':
    main()
