# -*- coding: utf-8 -*-
"""Model selection running script.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * include reformat.
    * skip some combinations.

"""
import os
import sys
import pickle
import logging
import argparse
import itertools

import pandas as pd

from .configs import (CleaningConfig, PreprocessingConfig, GridSearchingConfig,
                      ModelSelectingConfig,
                      DefaultDecisionTreeClassifierConfig,
                      DefaultAdaBoostClassifierConfig,
                      DefaultRandomForestClassifierConfig,
                      DefaultMLPClassifierConfig)
from .cleaner import Cleaner
from .preprocessor import Preprocessor
from .model_selector import ModelSelector
from .utils.visualization import *
from .utils.analysis import *

logging.getLogger(__file__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG)


def load_X_and_y(
        path_data,
        col_dependent=ModelSelectingConfig.COLUMN_DEPENDENT,
        mode='pkl'):
    """
    """
    if mode == 'pkl':
        data = pd.read_pickle(path_data)
    elif mode == 'csv':
        data = pd.read_csv(path_data)
    else:
        raise ValueError(f"Invalid read mode: {mode}")

    y = data[col_dependent]
    bi_nan = y.isnull()  # Boolean index of missing values.

    X = data.drop([col_dependent], axis=1)[~bi_nan]
    y = y[~bi_nan]

    return X, y


def dump_X_and_y(X, y, path_data):
    """
    """
    data = pd.concat([X, y], axis=1)

    data.to_pickle(path_data)


def run_model_combination(
        X: pd.DataFrame,
        y: pd.Series,
        classifier_mode,
        scale_mode,
        impute_mode,
        outlier_mode,
        skip_preprocessing=False) -> any:  # TODO
    """
    """
    mgs = ModelSelector(
        classifier_mode=classifier_mode)

    if not skip_preprocessing:
        preprocessor = Preprocessor(
            scale_mode=scale_mode,
            impute_mode=impute_mode,
            outlier_mode=outlier_mode)
        X, y = preprocessor.preprocess(
            X, y, var_cat=PreprocessingConfig.columns_categorical)

        dump_X_and_y(
            X,
            y,
            ModelSelectingConfig.get_default_preprocessed_data_path(
                scale_mode, impute_mode, outlier_mode))

    if classifier_mode == 'decisiontreeclassifier':
        param_grid = DefaultDecisionTreeClassifierConfig.get_param_grid()
    elif classifier_mode == 'adaboostclassifier':
        param_grid = DefaultAdaBoostClassifierConfig.get_param_grid()
    elif classifier_mode == 'randomforestclassifier':
        param_grid = DefaultRandomForestClassifierConfig.get_param_grid()
    elif classifier_mode == 'mlpclassifier':
        param_grid = DefaultMLPClassifierConfig.get_param_grid()
    else:
        logging.info(f"Not performing grid search for {classifier_mode}.")
        best_score = mgs.run_model_cv(X, y)
        best_params = None

        return best_params, best_score

    best_score, best_params = mgs.run_model_grid_search_cv(
        X, y, param_grid)

    return best_params, best_score


def parse_args():
    """
    TODO
        output file path and err path.
    """
    parser = argparse.ArgumentParser(
        description="Perform model selection for ALSPAC dataset.")
    parser.add_argument('--datapath', type=argparse.FileType('r'),
                        default=ModelSelectingConfig.PATH_DATA_INPUT_FILE,
                        help="Input data file path.")
    args = parser.parse_args(sys.argv[1:])

    return args


def main():
    """Initialize settings and perform model selection while recording results.

    """
    logging.info(
        ("Initiating model selection...\n"
         f"Target variable: {ModelSelectingConfig.COLUMN_DEPENDENT}"))

    #test_data_clean_path = os.path.abspath(os.path.dirname(__file__)) + \
    #    "/output/test_data_clean.csv"
    test_data_clean_path = ModelSelectingConfig.PATH_DATA_CLEANED
    try:  # Load cleaned data if exists.
        X, y = load_X_and_y(test_data_clean_path)
    except Exception:  # Generate cleaned data and store.
        logging.warning(
            ("Cleaned data pickle file not found. Generating and storing "
             f"cleaned data at {test_data_clean_path}"))

        # Load raw data.
        path_data_raw = parse_args().datapath.name
        X, y = load_X_and_y(path_data_raw, mode='csv')

        # Perform data cleaning.
        logging.info(
            ("Performing data cleaning...\n"
             f"Ignoring columns: {CleaningConfig.COLUMNS_IGNORED}\n"
             "Droping columns with missing value ratios greater "
             f"than {CleaningConfig.THRESHOLD_DROP_MISSING_RATIO}"))
        cleaner = Cleaner(
            cols_ignored=CleaningConfig.COLUMNS_IGNORED,
            thres_mis=CleaningConfig.THRESHOLD_DROP_MISSING_RATIO)
        X = cleaner.clean(X, ModelSelectingConfig.AGE_CUTOFF)
        logging.info(f"Cleaned training data shape: {X.shape}")

        logging.info(
            (f"Dumping cleaned data to "
             f"{test_data_clean_path}"))
        dump_X_and_y(X, y, test_data_clean_path)

    path_data_raw = parse_args().datapath.name
    # drops data missing dependent variable
    X_orig, y_orig = load_X_and_y(path_data_raw, mode='csv')
    original_data = pd.concat([X_orig, y_orig], axis=1)
    logging.info(f"Data shape: {original_data.shape}")
    missing_value_path = os.path.abspath(os.path.dirname(__file__)) + \
        "/output/missing_value.png"
    plot_missing_value_ratio_histogram(original_data, missing_value_path)
    # graph for raw data without dropping rows that do not have dependent variable
    missing_value_path = os.path.abspath(os.path.dirname(__file__)) + \
        "/output/missing_value_original_no_drop_dependent.png"
    # keep dependent variable
    data = pd.read_csv(ModelSelectingConfig.PATH_DATA_INPUT_FILE)
    plot_missing_value_ratio_histogram(data, missing_value_path)
    # code from visualization to get data info for chart
    ratio_missing = data.isnull().sum() / len(data) # find ratio missing to put in csv
    ax = ratio_missing.plot.hist(
        bins=10,
        alpha=0.5,
        title="Missing Value Ratio Histogram")
    ax.set_xlabel("Ratio")
    p = ax.patches
    heights = [path.get_height() for path in p]
    logging.info(f"Heights from histogram raw: {heights}")
    # code to put missing values info of data that is uncleaned but did
    # drop data without dependent variable
    ratio_missing_drop_dep = (original_data.isnull().sum()/len(original_data)).to_frame().reset_index()
    #logging.info(f"{ratio_missing_drop_dep}")
    ratio_missing_drop_dep = ratio_missing_drop_dep.set_axis(["Variable_Name", "Missing"], axis=1)
    missing_value_path = os.path.abspath(os.path.dirname(__file__)) + \
        "/output/missing_value_original_drop_dependent.csv"
    # sort by second column
    ratio_missing_drop_dep = ratio_missing_drop_dep.sort_values(
        ratio_missing_drop_dep.columns[1], ascending=False)
    ratio_missing_drop_dep.to_csv(missing_value_path, index=False)



if __name__ == '__main__':
    main()
