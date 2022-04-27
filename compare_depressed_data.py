# -*- coding: utf-8 -*-
"""Compare what data exists at what indices

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * 

python -u -m DepressionProjectNew.compare_depressed_data \
./DepressionProjectNew/output/preprocessed_data_without_temporal.txt \
y12CH_Dep_YN_144m \
y16CH_Dep_YN_192m \
y17CH_Dep_YN_204m \
y18CH_Dep_YN_216m
"""

import os
import sys
import pickle
import logging
import argparse

import numpy as np
import pandas as pd
import click

from .configs import (CleaningConfig, PreprocessingConfig, GridSearchingConfig,
                      ModelSelectingConfig,
                      DefaultDecisionTreeClassifierConfig,
                      DefaultAdaBoostClassifierConfig,
                      DefaultRandomForestClassifierConfig,
                      DefaultMLPClassifierConfig)
from msap.modeling.configs import (
    GridSearchConfig,
    ModelSelectionConfig)
from msap.modeling.model_selection.preprocessing import Preprocessor
from msap.utils import (
    ClassifierHandler,
    load_X_and_y,
    dump_X_and_y,
    KFold_by_feature)

logging.getLogger(__file__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG)

@click.command()
@click.argument(
    'path-data',
    type=str)
@click.argument(
    'feature-label-12',
    type=str)
@click.argument(
    'feature-label-16',
    type=str)
@click.argument(
    'feature-label-17',
    type=str)
@click.argument(
    'feature-label-18',
    type=str)
def main(
        path_data,
        feature_label_12,
        feature_label_16,
        feature_label_17,
        feature_label_18):

    # load data that has been preprocessed but not cleaned
    data = pd.read_csv(path_data)

    logging.info(f"Orig Data Shape: {data.shape}")

    # compare data
    has_dep = data[data[feature_label_12].notna() & \
        data[feature_label_16].notna() & \
        data[feature_label_17].notna() & \
        data[feature_label_18].notna()]
    
    logging.info(f"Has All Dep Shape: {has_dep.shape}")

    # Has 12 and 16
    has_12_16_dep = data[data[feature_label_12].notna() & \
        data[feature_label_16].notna()]
    
    logging.info(f"Has 12 and 16 Dep Shape: {has_12_16_dep.shape}")

    # Has 12 and 16 and 17
    has_12_16_17_dep = data[data[feature_label_12].notna() & \
        data[feature_label_16].notna() & \
        data[feature_label_17].notna()]
    
    logging.info(f"Has 12 and 16 and 17 Shape: {has_12_16_17_dep.shape}")

    # Has 18 check
    #has_18_dep = data[data[feature_label_18].notna()]
    
    #logging.info(f"Has 18 Dep Shape: {has_18_dep.shape}")

    # check how much missing data in each column
    mvr_df = data.isnull().mean(axis=1)
    #logging.info(f"Missing Info by Row: {mvr_df}")
    data_removed_df = data.loc[mvr_df[mvr_df >= 0.1].index]
    logging.info(f"Removed Samples with over 10% missing data by row: {data_removed_df.shape}")
    data_new_df = data.loc[mvr_df[mvr_df < 0.1].index]
    logging.info(f"Shape: {data_new_df.shape}")
    logging.info(f"Age 12 Depressed of data with over 10% missing by"
        f" row:\n{data_new_df[feature_label_12].value_counts()}")
    logging.info(f"Age 16 Depressed of data with over 10% missing by"
        f" row:\n{data_new_df[feature_label_16].value_counts()}")
    logging.info(f"Age 17 Depressed of data with over 10% missing by"
        f" row:\n{data_new_df[feature_label_17].value_counts()}")
    logging.info(f"Age 18 Depressed of data with over 10% missing by"
        f" row:\n{data_new_df[feature_label_18].value_counts()}")
    

    data_removed_df = data.loc[mvr_df[mvr_df >= 0.15].index]
    logging.info(f"Removed Samples with over 15% missing data by row: {data_removed_df.shape}")
    data_new_df = data.loc[mvr_df[mvr_df < 0.15].index]
    logging.info(f"Shape: {data_new_df.shape}")
    logging.info(f"Age 12 Depressed of data with over 15% missing by"
        f" row:\n{data_new_df[feature_label_12].value_counts()}")
    logging.info(f"Age 16 Depressed of data with over 15% missing by"
        f" row:\n{data_new_df[feature_label_16].value_counts()}")
    logging.info(f"Age 17 Depressed of data with over 15% missing by"
        f" row:\n{data_new_df[feature_label_17].value_counts()}")
    logging.info(f"Age 18 Depressed of data with over 15% missing by"
        f" row:\n{data_new_df[feature_label_18].value_counts()}")
    
    data_removed_df = data.loc[mvr_df[mvr_df >= 0.20].index]
    logging.info(f"Removed Samples with over 20% missing data by row: {data_removed_df.shape}")
    data_new_df = data.loc[mvr_df[mvr_df < 0.20].index]
    logging.info(f"Shape: {data_new_df.shape}")
    logging.info(f"Age 12 Depressed of data with over 20% missing by"
        f" row:\n{data_new_df[feature_label_12].value_counts()}")
    logging.info(f"Age 16 Depressed of data with over 20% missing by"
        f" row:\n{data_new_df[feature_label_16].value_counts()}")
    logging.info(f"Age 17 Depressed of data with over 20% missing by"
        f" row:\n{data_new_df[feature_label_17].value_counts()}")
    logging.info(f"Age 18 Depressed of data with over 20% missing by"
        f" row:\n{data_new_df[feature_label_18].value_counts()}")
    



if __name__ == '__main__':
    main()
