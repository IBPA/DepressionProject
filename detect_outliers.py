# -*- coding: utf-8 -*-
"""Outlier detection methods.

Authors:
    Jason Youn - jyoun@ucdavis.edu
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * Docstring

"""
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# changed outlier detectors to n_jobs = 1
def convert_index_2_bool(indices):
    """Convert integer style outlier / inlier indices to boolean.

    Args:
        indices (list): -1 for outliers and 1 for inliers.

    Returns:
        (list): False for outliers and True for inliers.

    """
    return [True if i == 1 else False for i in indices]


def isolation_forest(data_df, random_state=None):
    """Detect outliers using the Isolation Forest algorithm.

    Args:
        data_df (pd.DataFrame): Input data.
        random_state (int, optional): Seed of the pseudo
            random number generator to use.

    Returns:
        (list): False for outliers and True for inliers.

    """

    data_array = data_df.to_numpy()
    clf = IsolationForest(
        n_jobs=1, contamination=0.05, random_state=random_state, verbose=5)
    clf.fit(data_array)

    return convert_index_2_bool(clf.predict(data_array).tolist())


def local_outlier_factor(data_df):
    """Detect outliers using the LOF algorithm.

    Args:
        data_df (pd.DataFrame): Input data.

    Returns:
        (list): False for outliers and True for inliers.

    """
    clf = LocalOutlierFactor(n_jobs=1)

    return convert_index_2_bool(clf.fit_predict(data_df.to_numpy()).tolist())
