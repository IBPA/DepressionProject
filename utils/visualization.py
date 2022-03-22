# -*- coding: utf-8 -*-
"""A one line summary.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * Docstring

"""
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .data import load_X_and_y

def plot_missing_value_ratio_histogram(data, path_save=None):
    """
    """
    ratio_missing = data.isnull().sum() / len(data)
    ax = ratio_missing.plot.hist(
        bins=10,
        alpha=0.5,
        title="Missing Value Ratio Histogram")
    ax.set_xlabel("Ratio")

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()


def plot_pairwise_pearson_correlation(X, y, path_save=None):
    """
    """
    corr_mat = pd.concat([X, y], axis=1).corr()
    sns.heatmap(corr_mat, cmap=plt.cm.bwr)
    plt.title("Pairwise Pearson Correlation")

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()

def plot_spearman_correlation(X, y, path_save=None):
    """
    """
    corr_mat = pd.concat([X, y], axis=1).corr(method = 'spearman')
    sns.heatmap(corr_mat, cmap=plt.cm.bwr)
    plt.title("Spearman Correlation")

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()

def plot_pca(X, y, path_save=None):
    """
    """
    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X.to_numpy())
    y = y.to_numpy()

    # Start plotting.
    labels = [1, 0]
    colors = ['red', 'blue']

    for label, color in zip(labels, colors):
        plt.scatter(
            x=[x[0] for i, x in enumerate(X_pca) if y[i] == label],
            y=[x[1] for i, x in enumerate(X_pca) if y[i] == label],
            c=color,
            label="Depressed" if label == 1 else "Normal",
            alpha=0.5)
    plt.legend(loc=4, numpoints=1)
    plt.title("PCA")
    plt.xlabel("First principal Component")
    plt.ylabel("Second principal Component")

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()
    
    X_pca_df = pd.DataFrame(X_pca, columns = ['component1', 'component2'])
    y = pd.DataFrame(y, columns = ['y'])
    return X_pca_df, y

def plot_tsne(X, y, path_save=None):
    """
    """
    tsne = TSNE(n_components=2)

    X_tsne = tsne.fit_transform(X)
    y = y.to_numpy()

    # Start plotting.
    labels = [1, 0]
    colors = ['red', 'blue']

    for label, color in zip(labels, colors):
        plt.scatter(
            x=[x[0] for i, x in enumerate(X_tsne) if y[i] == label],
            y=[x[1] for i, x in enumerate(X_tsne) if y[i] == label],
            c=color,
            label="Depressed" if label == 1 else "Normal",
            alpha=0.5)
    plt.legend(loc=4, numpoints=1)
    plt.title("tSNE")
    plt.xlabel("First Embedding")
    plt.ylabel("Second Embedding")

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()

    X_tsne_df = pd.DataFrame(X_tsne, columns = ['component1', 'component2'])
    y = pd.DataFrame(y, columns = ['y'])
    return X_tsne_df, y

def plot_sfs(metric_dict,
             xticks,
             title,
             path_save='sfs.png',
             kind='std_dev',
             color='blue',
             bcolor='steelblue',
             marker='o',
             alpha=0.2,
             confidence_interval=0.95):
    import os

    allowed = {'std_dev', 'std_err', 'ci', None}
    if kind not in allowed:
        raise AttributeError('kind not in %s' % allowed)

    plt.figure()

    k_feat = sorted(metric_dict.keys())
    avg = [metric_dict[k]['avg_score'] for k in k_feat]

    if kind:
        upper, lower = [], []
        if kind == 'ci':
            kind = 'ci_bound'

        for k in k_feat:
            upper.append(metric_dict[k]['avg_score'] +
                         metric_dict[k][kind])
            lower.append(metric_dict[k]['avg_score'] -
                         metric_dict[k][kind])

        plt.fill_between(k_feat,
                         upper,
                         lower,
                         alpha=alpha,
                         color=bcolor,
                         lw=1)

        if kind == 'ci_bound':
            kind = 'Confidence Interval (%d%%)' % (confidence_interval * 100)

    plt.plot(k_feat, avg, color=color, marker=marker)
    plt.xlabel('Features')
    plt.ylabel('F1-score')
    plt.title(title)
    plt.grid()

    plt.xticks(range(1, len(xticks)+1), xticks,
        rotation=90, fontsize=10)
    
    # using
    # https://stackoverflow.com/questions/44863375/how-to-change-spacing-between-ticks-in-matplotlib
    plt.gca().margins(x=0)
    plt.gcf().canvas.draw()
    ticks = plt.gca().get_xticklabels()
    maxsize = 20 # tested diff constants
    m = 0.3 # tested diff constants
    s = maxsize/plt.gcf().dpi*len(xticks)+2*m
    margin = m/plt.gcf().get_size_inches()[0]

    if path_save is not None:
        plt.gcf().subplots_adjust(left=margin, right=1.-margin)
        plt.gcf().set_size_inches(s, plt.gcf().get_size_inches()[1])
        plt.savefig(path_save, bbox_inches='tight')
    plt.close()

def plot_pr_curve(metrics, path_save=None):
    """
    """
    plt.figure(figsize=(20, 20))

    lines = []
    labels = []

    for classifier, results in metrics.items():
        if classifier == 'dummyclassifier':  # Baseline.
            classifier = 'baseline'
            linestyle = '--'
            precision = results['pr_curve'][0]
            precision[1] = precision[0]
            recall = results['pr_curve'][1]
        else:
            linestyle = '-'
            precision = results['pr_curve'][0]
            recall = results['pr_curve'][1]
        line, = plt.plot(
            recall, precision, linestyle=linestyle, linewidth=3)
        label = f"{classifier} (AUCPR = {results['pr_auc']: .2f})"
        lines.append(line)
        labels.append(label)

    plt.title("Precision-recall for Depressed", fontsize=30)
    plt.xlabel("Recall", fontsize=30)
    plt.ylabel("Precision", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(lines, labels, loc=3, fontsize=24)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()


def plot_roc_curve(metrics, path_save=None):
    """
    """
    plt.figure(figsize=(20, 20))

    lines = []
    labels = []

    for classifier, results in metrics.items():
        if classifier == 'dummyclassifier':  # Baseline.
            classifier = 'baseline'
            linestyle = '--'
            fpr = results['roc_curve'][0]
            tpr = results['roc_curve'][1]
        else:
            linestyle = '-'
            fpr = results['roc_curve'][0]
            tpr = results['roc_curve'][1]
        line, = plt.plot(
            fpr, tpr, linestyle=linestyle, linewidth=3)
        label = f"{classifier} (AUCROC = {results['roc_auc']: .2f})"
        lines.append(line)
        labels.append(label)

    plt.title("Receiver Operating Characteristic for Depressed", fontsize=30)
    plt.xlabel("False Positive Rate", fontsize=30)
    plt.ylabel("True Positive Rate", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(lines, labels, loc=4, fontsize=24)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    if path_save is not None:
        plt.savefig(path_save)
    plt.close()


if __name__ == '__main__':
    import os
    import pickle

    path_output_dir = (
        os.path.abspath(os.path.dirname(__file__))
        + "/../output")

    metrics = pickle.load(
        open(path_output_dir + "/analyses/metrics.pkl", 'rb'))

    plot_pr_curve(metrics, path_output_dir + "/figs/pr_curve.png")
    plot_roc_curve(metrics, path_output_dir + "/figs/roc_curve.png")
    plot_pr_curve(
        {'adaboostclassifier': metrics['adaboostclassifier']},
        path_output_dir + "/figs/pr_curve_best.png")
