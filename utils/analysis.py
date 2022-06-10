"""
Authors:
    Jason

TODO:
    comments

"""

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import shapiro
from sklearn.metrics import (confusion_matrix,
                             precision_recall_curve,
                             average_precision_score,
                             roc_curve,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import datetime
import logging

from .visualization import *


def analyze_pearson_correlation(
        X: pd.DataFrame,
        y: pd.Series,
        path_save: str = None) -> None:
    """Analyze PCC and p-values.

    Args:
        X: Input data.
        y: Target data.
        path_save (optional): The location where Pearson correlation analysis
            result stores.

    Generates:
        (optional): A file storing pearson correlation results.

    """
    pearson_result = [pearsonr(X[feature], y) for feature in list(X)]

    scores = list(zip(*pearson_result))[0]
    scores_abs = [abs(s) for s in scores]
    pvalues = list(zip(*pearson_result))[1]

    indices = np.argsort(scores_abs)[::-1]
    ranked = [list(X)[idx] for idx in indices]

    res = []
    for i in range(X.shape[1]):
        res.append(
            [i + 1, ranked[i], scores[indices[i]], pvalues[indices[i]]])
    res_df = pd.DataFrame(res)
    res_df.columns = ['Rank', 'Variable', 'Correlation', 'P-value']

    if path_save is not None:
        res_df.to_csv(path_save)
    else:
        # Log the feature ranking.
        logging.debug('Pairwise feature pearson correlation ranking:')
        for f in range(X.shape[1]):
            logging.debug(
                '%d. %s: %f (%f)',
                f + 1,
                ranked[f],
                scores[indices[f]],
                pvalues[indices[f]])

    return res_df


def analyze_spearman_correlation(
        X: pd.DataFrame,
        y: pd.Series,
        path_save: str = None) -> None:
    # TODO - test
    """Analyze spearman and p-values.

    Args:
        X: Input data.
        y: Target data.
        path_save (optional): The location where Spearman correlation analysis
            result stores.

    Generates:
        (optional): A file storing spearman correlation results.

    """
    spearman_result = [spearmanr(X[feature], y) for feature in list(X)]

    scores = list(zip(*spearman_result))[0]
    scores_abs = [abs(s) for s in scores]
    pvalues = list(zip(*spearman_result))[1]

    indices = np.argsort(scores_abs)[::-1]
    ranked = [list(X)[idx] for idx in indices]

    res = []
    for i in range(X.shape[1]):
        res.append(
            [i + 1, ranked[i], scores[indices[i]], pvalues[indices[i]]])
    res_df = pd.DataFrame(res)
    res_df.columns = ['Rank', 'Variable', 'Correlation', 'P-value']

    if path_save is not None:
        res_df.to_csv(path_save)
    else:
        # Log the feature ranking.
        logging.debug('Pairwise feature spearman correlation ranking:')
        for f in range(X.shape[1]):
            logging.debug(
                '%d. %s: %f (%f)',
                f + 1,
                ranked[f],
                scores[indices[f]],
                pvalues[indices[f]])

    return res_df


"""
def perform_pca(X, y, path_save = None):
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X.to_numpy())

    X_pca_df = pd.DataFrame(X_pca, columns = ['component1', 'component2'])

    y = y.to_numpy()
    y = pd.DataFrame(y, columns = ['y'])

    if path_save is not None:
        data = pd.concat([X_pca_df, y], axis=1)
        data.to_csv(path_save)

    return X_pca, y

def perform_tsne(X, y, path_save = None):
    tsne = TSNE(n_components=2)

    X_tsne = tsne.fit_transform(X.to_numpy())

    X_tsne_df = pd.DataFrame(X_tsne, columns = ['component1', 'component2'])

    y = y.to_numpy()
    y = pd.DataFrame(y, columns = ['y'])

    if path_save is not None:
        data = pd.concat([X_tsne_df, y], axis=1)
        data.to_csv(path_save)
    
    return X_tsne, y
"""


def analyze_normal_distribution(X, path_save):
    # TODO - doesn't work
    df = pd.DataFrame()
    df['Variable'] = X.columns
    #logging.info(f"X: {X}")
    logging.info(f"First Shapiro {X.columns[0]}: {shapiro(X[X.columns[0]])}")
    #df['Shapiro'] = X.apply(lambda x: shapiro(x).statistic, axis=1)
    #df['P-value'] = X.apply(lambda x: shapiro(x).pvalue, axis=1)
    shap = []
    pval = []
    for col_name, col_data in X.iteritems():
        shap.append(shapiro(col_data).statistic)
        pval.append(shapiro(col_data).pvalue)
    df['Shapiro'] = shap
    df['P-value'] = pval
    if path_save is not None:
        df.to_csv(path_save)
    return df


def analyze_pca(X, y, pearson_df=None, spearman_df=None, path_dir=None, rfe_seed=None):
    num_rows = 20
    if pearson_df is not None:
        pearson_df.sort_values(
            by='Correlation', ascending=False, ignore_index=True, key=abs)
        pearson_df = pearson_df[(pearson_df['P-value'] <= 0.05)
                                & (pearson_df['P-value'] != 0)]
        if num_rows <= len(pearson_df):
            top_pearson = pearson_df.head(num_rows)
        else:
            top_pearson = pearson_df
        X_tp = top_pearson["Variable"].tolist()
        if path_dir is not None:
            ct = datetime.datetime.now()
            if rfe_seed is None:
                X_pca, y = plot_pca(
                    X[X_tp],
                    y,
                    path_dir
                    + "/pca_top20pearson" + str(ct) + ".png")
                data = pd.concat([X_pca, y], axis=1)
                data.to_csv(path_dir
                            + "/../analyses/pca_top20pearson" + str(ct) + ".csv")
                analyze_normal_distribution(X[X_tp], path_dir
                                            + "/../analyses/shapiro_top20pearson" + str(ct) + ".csv")
            else:
                X_pca, y = plot_pca(
                    X[X_tp],
                    y,
                    path_dir
                    + "/pca_top20pearson_rfeseed_" + str(rfe_seed) + "_"
                    + str(ct) + ".png")
                data = pd.concat([X_pca, y], axis=1)
                data.to_csv(path_dir
                            + "/../analyses/pca_top20pearson_rfeseed_" +
                            str(rfe_seed)
                            + "_" + str(ct) + ".csv")
                analyze_normal_distribution(X[X_tp], path_dir
                                            + "/../analyses/shapiro_top20pearson_rfeseed_"
                                            + str(rfe_seed) + "_" + str(ct) + ".csv")
        else:
            plot_pca(
                X[X_tp],
                y)

    if spearman_df is not None:
        spearman_df.sort_values(
            by='Correlation', ascending=False, ignore_index=True, key=abs)
        spearman_df = spearman_df[(
            spearman_df['P-value'] <= 0.05) & (spearman_df['P-value'] != 0)]
        if num_rows <= len(spearman_df):
            top_spearman = spearman_df.head(num_rows)
        else:
            top_spearman = spearman_df
        X_ts = top_spearman["Variable"].tolist()
        if path_dir is not None:
            ct = datetime.datetime.now()
            if rfe_seed is None:
                X_pca, y = plot_pca(
                    X[X_ts],
                    y,
                    path_dir
                    + "/pca_top20spearman" + str(ct) + ".png")
                data = pd.concat([X_pca, y], axis=1)
                data.to_csv(path_dir
                            + "/../analyses/pca_top20spearman" + str(ct) + ".csv")
                analyze_normal_distribution(X[X_ts], path_dir
                                            + "/../analyses/shapiro_top20spearman" + str(ct) + ".csv")
            else:
                X_pca, y = plot_pca(
                    X[X_ts],
                    y,
                    path_dir
                    + "/pca_top20spearman_rfeseed_" + str(rfe_seed) + "_"
                    + str(ct) + ".png")
                data = pd.concat([X_pca, y], axis=1)
                data.to_csv(path_dir
                            + "/../analyses/pca_top20spearman_rfeseed_" +
                            str(rfe_seed)
                            + "_" + str(ct) + ".csv")
                analyze_normal_distribution(X[X_ts], path_dir
                                            + "/../analyses/shapiro_top20spearman_rfeseed_"
                                            + str(rfe_seed) + "_" + str(ct) + ".csv")
        else:
            plot_pca(
                X[X_ts],
                y)

    if path_dir is not None:
        ct = datetime.datetime.now()
        if rfe_seed is None:
            X_pca, y = plot_pca(
                X,
                y,
                path_dir
                + "/pca_all" + str(ct) + ".png")
            data = pd.concat([X_pca, y], axis=1)
            data.to_csv(path_dir
                        + "/../analyses/pca_all" + str(ct) + ".csv")
        else:
            X_pca, y = plot_pca(
                X,
                y,
                path_dir
                + "/pca_all_rfeseed_" + str(rfe_seed)
                + "_" + str(ct) + ".png")
            data = pd.concat([X_pca, y], axis=1)
            data.to_csv(path_dir
                        + "/../analyses/pca_all_rfeseed_" + str(rfe_seed)
                        + "_" + str(ct) + ".csv")

    else:
        plot_pca(
            X,
            y)
    return


def analyze_tsne(X, y, pearson_df=None, spearman_df=None, path_dir=None, rfe_seed=None):
    num_rows = 20
    if pearson_df is not None:
        pearson_df.sort_values(
            by='Correlation', ascending=False, ignore_index=True, key=abs)
        pearson_df = pearson_df[(pearson_df['P-value'] <= 0.05)
                                & (pearson_df['P-value'] != 0)]
        if num_rows <= len(pearson_df):
            top_pearson = pearson_df.head(num_rows)
        else:
            top_pearson = pearson_df
        #logging.info(f"Top Pearson: {top_pearson['Correlation']}")
        X_tp = top_pearson["Variable"].tolist()
        if path_dir is not None:
            ct = datetime.datetime.now()
            if rfe_seed is None:
                X_tsne, y = plot_tsne(
                    X[X_tp],
                    y,
                    path_dir
                    + "/tsne_top20pearson" + str(ct) + ".png")
                data = pd.concat([X_tsne, y], axis=1)
                data.to_csv(path_dir
                            + "/../analyses/tsne_top20pearson" + str(ct) + ".csv")
                analyze_normal_distribution(X[X_tp], path_dir
                                            + "/../analyses/shapiro_top20pearson" + str(ct) + ".csv")
            else:
                X_tsne, y = plot_tsne(
                    X[X_tp],
                    y,
                    path_dir
                    + "/tsne_top20pearson_rfeseed_" + str(rfe_seed)
                    + "_" + str(ct) + ".png")
                data = pd.concat([X_tsne, y], axis=1)
                data.to_csv(path_dir
                            + "/../analyses/tsne_top20pearson_rfeseed_" +
                            str(rfe_seed)
                            + "_" + str(ct) + ".csv")
                analyze_normal_distribution(X[X_tp], path_dir
                                            + "/../analyses/shapiro_top20pearson_rfeseed_" +
                                            str(rfe_seed)
                                            + "_" + str(ct) + ".csv")
        else:
            plot_tsne(
                X[X_tp],
                y)

    if spearman_df is not None:
        spearman_df.sort_values(
            by='Correlation', ascending=False, ignore_index=True, key=abs)
        spearman_df = spearman_df[(
            spearman_df['P-value'] <= 0.05) & (spearman_df['P-value'] != 0)]
        if num_rows <= len(spearman_df):
            top_spearman = spearman_df.head(num_rows)
        else:
            top_spearman = spearman_df
        X_ts = top_spearman["Variable"].tolist()
        if path_dir is not None:
            ct = datetime.datetime.now()
            if rfe_seed is None:
                X_tsne, y = plot_tsne(
                    X[X_ts],
                    y,
                    path_dir
                    + "/tsne_top20spearman" + str(ct) + ".png")
                data = pd.concat([X_tsne, y], axis=1)
                data.to_csv(path_dir
                            + "/../analyses/tsne_top20spearman" + str(ct) + ".csv")
                analyze_normal_distribution(X[X_ts], path_dir
                                            + "/../analyses/shapiro_top20spearman" + str(ct) + ".csv")
            else:
                X_tsne, y = plot_tsne(
                    X[X_ts],
                    y,
                    path_dir
                    + "/tsne_top20spearman_rfeseed_" + str(rfe_seed)
                    + "_" + str(ct) + ".png")
                data = pd.concat([X_tsne, y], axis=1)
                data.to_csv(path_dir
                            + "/../analyses/tsne_top20spearman_rfeseed_"
                            + str(rfe_seed)
                            + "_" + str(ct) + ".csv")
                analyze_normal_distribution(X[X_ts], path_dir
                                            + "/../analyses/shapiro_top20spearman_rfeseed_"
                                            + str(rfe_seed)
                                            + "_" + str(ct) + ".csv")
        else:
            plot_tsne(
                X[X_ts],
                y)

    if path_dir is not None:
        ct = datetime.datetime.now()
        if rfe_seed is None:
            X_tsne, y = plot_tsne(
                X,
                y,
                path_dir
                + "/tsne_all" + str(ct) + ".png")
            data = pd.concat([X_tsne, y], axis=1)
            data.to_csv(path_dir
                        + "/../analyses/tsne_all" + str(ct) + ".csv")
        else:
            X_tsne, y = plot_tsne(
                X,
                y,
                path_dir
                + "/tsne_all_rfeseed_"
                + str(rfe_seed) + "_" + str(ct) + ".png")
            data = pd.concat([X_tsne, y], axis=1)
            data.to_csv(path_dir
                        + "/../analyses/tsne_all_rfeseed_"
                        + str(rfe_seed) + "_" + str(ct) + ".csv")
    else:
        plot_tsne(
            X,
            y)

    return


def analyze_elbox_method():
    return


def analyze_bic():
    return


def analyze_aic():
    return


def analyze_gap_statistic():
    return


def analyze_kendall_tau():
    return


def get_evaluation_curves(clf, X, y, curves=['pr', 'roc'], random_state=None):
    """
    """
    res = {}
    y_test_all = []
    y_pred_proba_all = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True,
                            random_state=random_state)

    for i_train, i_test in kfold.split(X):
        X_train, X_test = X.iloc[i_train].to_numpy(), X.iloc[i_test].to_numpy()
        y_train, y_test = y.iloc[i_train].to_numpy(), y.iloc[i_test].to_numpy()

        clf.fit(X_train, y_train)
        y_pred_proba_all.extend(clf.predict_proba(X_test)[:, 1])
        y_test_all.extend(y_test)

    for curve in curves:
        if curve == 'pr':
            precision, recall, _ = precision_recall_curve(
                y_test_all, y_pred_proba_all)
            res['pr'] = {}
            res['pr']['precision'] = precision
            res['pr']['recall'] = recall
            res['pr']['map'] = average_precision_score(
                y_test_all, y_pred_proba_all)
        elif curve == 'roc':
            fpr, tpr, _ = roc_curve(y_test_all, y_pred_proba_all)
            res['roc'] = {}
            res['roc']['fpr'] = fpr
            res['roc']['tpr'] = tpr
            res['roc']['auroc'] = roc_auc_score(y_test_all, y_pred_proba_all)

    return res


def get_cv_evaluation_values(
        clf: any,
        X: pd.DataFrame,
        y: pd.Series,
        scoring=[
            'accuracy', 'precision', 'recall', 'average_precision', 'roc_auc',
            'f1'],
        random_state=None):
    """Generate CV results using specifier metrics for the classifier.

    Args:
        clf: A classifier object implementing `fit`.
        X: Input data.
        y: Target data.
        metrics: A list of metrics for evaluation.

    Returns:
        None

    """
    def run_one_fold(clf, X, y, i_train, i_test):
        """
        """
        X_train, X_test = X.iloc[i_train].to_numpy(), X.iloc[i_test].to_numpy()
        y_train, y_test = y.iloc[i_train].to_numpy(), y.iloc[i_test].to_numpy()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        specificity = tn / (tn + fp)
        npv = tn / (tn + fn)
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        f1 = 2 * precision * recall / (precision + recall)

        return {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'nvp': npv,
            'accuracy': accuracy,
            'f1': f1,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba}

    # Parallelize CV to get evaluation metric values.
    kfold = StratifiedKFold(n_splits=5, shuffle=True,
                            random_state=random_state)
    cv_results = pd.DataFrame(Parallel(n_jobs=-1)(
        delayed(run_one_fold)(clf, X, y, i_train, i_test)
        for i_train, i_test in kfold.split(X)))

    # Calculate mean/std for each scalar metric.
    res = {}
    for metric in cv_results.columns[:-2]:
        res[metric + '_mean'] = np.mean(cv_results[metric])
        res[metric + '_std'] = np.std(cv_results[metric])

    # Calculate evaluation curves.
    y_test_all = np.concatenate(cv_results['y_test'].to_numpy())
    y_pred_proba_all = np.concatenate(cv_results['y_pred_proba'].to_numpy())
    res['pr_curve'] = precision_recall_curve(
        y_test_all, y_pred_proba_all)[:2]
    res['pr_auc'] = average_precision_score(
        y_test_all, y_pred_proba_all)
    res['roc_curve'] = roc_curve(
        y_test_all, y_pred_proba_all)[:2]
    res['roc_auc'] = roc_auc_score(
        y_test_all, y_pred_proba_all)

    return res


if __name__ == '__main__':
    import os
    import pickle

    from .data import load_X_and_y
    from ..classifier_handler import ClassifierHandler
    from ..configs.best_model import (BestDummyClassifierConfig,
                                      BestDecisionTreeClassifierConfig,
                                      BestGaussianNBConfig,
                                      BestMultinomialNBConfig,
                                      BestSVCConfig,
                                      BestAdaBoostClassifierConfig,
                                      BestRandomForestClassifierConfig,
                                      BestMLPClassifierConfig)
    from ..configs import ModelSelectingConfig

    res = {}

    # Load all best model parameters.
    model_params = []
    model_params.append(BestDummyClassifierConfig.get_params())
    model_params.append(BestDecisionTreeClassifierConfig.get_params())
    model_params.append(BestGaussianNBConfig.get_params())
    model_params.append(BestMultinomialNBConfig.get_params())
    model_params.append(BestSVCConfig.get_params())
    model_params.append(BestAdaBoostClassifierConfig.get_params())
    model_params.append(BestRandomForestClassifierConfig.get_params())
    model_params.append(BestMLPClassifierConfig.get_params())

    for model_param in model_params:
        random_state = 42
        if model_param['params'] is not None and 'random_state' in model_param['params']:
            random_state = model_param['params']['random_state']
        X, y = load_X_and_y(
            ModelSelectingConfig.get_default_preprocessed_data_path(
                model_param['scale_mode'],
                model_param['impute_mode'],
                model_param['outlier_mode'],
                random_state))

        clf = ClassifierHandler(
            classifier_mode=model_param['classifier_mode'],
            params=model_param['params']).clf

        res[model_param['classifier_mode']]\
            = get_cv_evaluation_values(clf, X, y)

    path_output_dir = (
        os.path.abspath(os.path.dirname(__file__))
        + "/../output")

    pickle.dump(res, open(path_output_dir + "/analyses/metrics.pkl", 'wb'))
