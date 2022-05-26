from ast import literal_eval

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_rfe_line_from_dataframe(
        sfs_result: pd.DataFrame,
        k_selected: int,
        title: str = None,
        path_save: str = None):
    """
    """
    features = sfs_result.loc[0, 'feature_names']
    feature_idx_list = sfs_result['feature_idx'].tolist()
    xticklabels = []
    for i in range(len(feature_idx_list) - 1, -1, -1):
        if i == len(feature_idx_list) - 1:
            xticklabels += [features[feature_idx_list[i][0]]]
        else:
            xticklabels += [
                features[
                    (set(feature_idx_list[i])
                     - set(feature_idx_list[i + 1])).pop()
                ]
            ]

    data_plot = []

    def unstack_cv_scores(data_plot, row):
        """
        """
        for score in row['cv_scores']:
            data_plot += [{'n_features': row['index'], 'score': score}]

    sfs_result.apply(
        lambda row: unstack_cv_scores(data_plot, row),
        axis=1)
    data_plot = pd.DataFrame(data_plot)
    xs = sfs_result['index'].tolist()

    # Overview plot.
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.lineplot(
        data=data_plot,
        x='n_features',
        y='score',
        ci='sd',
        ax=axs[0],
    )
    axs[0].vlines(
        k_selected,
        axs[0].get_ylim()[0],
        axs[0].get_ylim()[1],
        linestyles='dashed',
        label='Best')
    axs[0].set_xticks(
        [max(xs), min(xs)]
        + list(range(max(xs), min(xs), -int(len(xs) / min(5, len(xs)))))
        + [k_selected])
    axs[0].legend()
    axs[0].set_xlabel('The Number of Features')
    axs[0].set_ylabel('F1 Score')

    # Detailed plot.
    sns.lineplot(
        data=data_plot,
        x='n_features',
        y='score',
        ci='sd',
        ax=axs[1],
    )
    axs[1].vlines(
        k_selected,
        axs[1].get_ylim()[0],
        axs[1].get_ylim()[1],
        linestyles='dashed',
        label='Best')
    axs[1].set_xticks(
        ticks=list(range(1, len(xticklabels) + 1)),
        labels=xticklabels,
        rotation=90,
    )
    axs[1].legend()
    axs[1].set_xlim(
        [max(1, k_selected - 10), min(k_selected + 10, len(features))]
    )
    axs[1].set_xlabel('Features')
    axs[1].set_ylabel('F1 Score')

    if title is not None:
        plt.suptitle(title)

    if path_save is not None:
        fig.savefig(path_save)
        plt.close()
    else:
        plt.show()


def get_parsimonious(rfe_result):
    """
    """
    n_features = len(rfe_result.loc[0, 'feature_idx'])
    i_best = rfe_result['avg_score'].argmax()
    k_best = len(rfe_result.loc[i_best, 'feature_idx'])
    score_best = rfe_result.loc[i_best, 'avg_score']

    for k in range(n_features, 0, -1):
        if k >= k_best:
            continue

        if rfe_result.loc[n_features - k, 'avg_score'] \
                >= (score_best - rfe_result.loc[n_features - k, 'std_dev']
                    / len(rfe_result.loc[n_features - k, 'cv_scores'])):
            k_best = k
            score_best = rfe_result.loc[n_features - k, 'avg_score']

    return k_best, score_best


if __name__ == "__main__":
    rfe_result = pd.read_csv(
        # './output/old_results/10MVIout/output_12_yesmental/rfe_result.csv'
        './rfe_result.csv'
    )
    rfe_result['feature_idx'] = rfe_result['feature_idx'].apply(literal_eval)
    rfe_result['cv_scores'] = rfe_result['cv_scores'].apply(
        lambda x: np.fromstring(x[1:-1], dtype=float, sep=' ')
    )
    rfe_result['feature_names'] = rfe_result['feature_names'].apply(
        literal_eval
    )

    plot_rfe_line_from_dataframe(
        rfe_result,
        get_parsimonious(rfe_result)[0],
        title='Sequential Feature Selection',
        path_save="rfe.svg",
    )

    # plot_rfe_line_from_metric_dict(
    #     rfe_result,
    #     './output/old_results/10MVIout/output_12_yesmental/age12_10MVI.svg'
    # )
