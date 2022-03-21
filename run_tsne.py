# -*- coding: utf-8 -*-
"""Model selection running script.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * include reformat.
    * I don;t like preprocessor...
    * Help for clicks

python -u -m DepressionProjectNew.run_tsne
./DepressionProjectNew/output/output_18_yesmental/preprocessed
robust missforest lof y18CH_Dep_YN_216m
"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
import click

from msap.modeling.configs import (
    GridSearchConfig,
    ModelSelectionConfig)
from msap.modeling.model_selection.train import train_grid_search_cv, train_cv
from msap.modeling.model_selection.preprocessing import Preprocessor
from msap.utils import (
    ClassifierHandler,
    load_X_and_y,
    dump_X_and_y,
    KFold_by_feature)
from msap.utils.plot import (
    plot_tsne_outliers,
    plot_tsne)

os.environ["PYTHONWARNINGS"] = (
    "ignore::RuntimeWarning"
)

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.DEBUG)


@click.command()
@click.argument(
    'path-data-preprocessed-dir',
    type=str)
@click.argument(
    'scale-mode',
    type=str)
@click.argument(
    'impute-mode',
    type=str)
@click.argument(
    'outlier-mode',
    type=str)
@click.argument(
    'feature-label',
    type=str)
@click.option(
    '--random-state',
    type=int,
    default=42)
def main(
        path_data_preprocessed_dir,
        scale_mode,
        impute_mode,
        outlier_mode,
        feature_label,
        random_state):
    """
    """
    np.random.seed(random_state)

    cfg_model = ModelSelectionConfig

    filename_data_scale_impute = cfg_model.get_filename_scale_impute_data(
        scale_mode, impute_mode, outlier_mode)
    X_scale_impute, y = load_X_and_y(
        f"{path_data_preprocessed_dir}/{filename_data_scale_impute}",
        feature_label)

    filename_data_prep = cfg_model.get_filename_preprocessed_data(
        scale_mode, impute_mode, outlier_mode)
    data = pd.read_csv(f"{path_data_preprocessed_dir}/{filename_data_prep}")
    X_prep = data.drop([feature_label], axis=1)
    y_prep = data[feature_label]

    filename_outliers = cfg_model.get_filename_outliers(
        scale_mode, impute_mode, outlier_mode)
    idxs_outlier = np.loadtxt(f"{path_data_preprocessed_dir}/{filename_outliers}")

    filename_tsne = cfg_model.get_filename_tsne(
        scale_mode, impute_mode, outlier_mode)
    filename_outliers_tsne = cfg_model.get_filename_outliers_tsne(
        scale_mode, impute_mode, outlier_mode)
    plot_tsne_outliers(
        X = X_scale_impute,
        y = y,
        idxs_outlier = idxs_outlier,
        path_save = f"{path_data_preprocessed_dir}/"
        f"{filename_outliers_tsne}")
    plot_tsne( # removed outliers
        X = X_prep,
        y = y_prep,
        path_save = f"{path_data_preprocessed_dir}/"
        f"{filename_tsne}")
    
    idxs_inlier = [i for i in range(len(X_scale_impute)) if i not in idxs_outlier]
    X_scale_impute = X_scale_impute.iloc[idxs_inlier]
    y_se = y.iloc[idxs_inlier]
    dump_X_and_y(
        X=X_scale_impute,
        y=y_se,
        path_output_data = f"{path_data_preprocessed_dir}/"
        f"test_{filename_data_prep}")

if __name__ == '__main__':
    main()