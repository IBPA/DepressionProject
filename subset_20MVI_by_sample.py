# -*- coding: utf-8 -*-
"""Compare what data exists for what depression variables

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * 

python -u -m DepressionProjectNew.subset_20MVI_by_sample \
./DepressionProjectNew/output/preprocessed_data_without_temporal.txt \
y12CH_Dep_YN_144m \
y16CH_Dep_YN_192m \
y17CH_Dep_YN_204m \
y18CH_Dep_YN_216m \
./DepressionProjectNew/output/drop_samples_12_preprocessed_data_without_temporal.csv \
./DepressionProjectNew/output/drop_samples_16_preprocessed_data_without_temporal.csv \
./DepressionProjectNew/output/drop_samples_17_preprocessed_data_without_temporal.csv \
./DepressionProjectNew/output/drop_samples_18_preprocessed_data_without_temporal.csv
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
@click.argument(
    'output-file-12',
    type=str)
@click.argument(
    'output-file-16',
    type=str)
@click.argument(
    'output-file-17',
    type=str)
@click.argument(
    'output-file-18',
    type=str)
def main(
        path_data,
        feature_label_12,
        feature_label_16,
        feature_label_17,
        feature_label_18,
        output_file_12,
        output_file_16,
        output_file_17,
        output_file_18):

    # load data that has been preprocessed but not cleaned
    data = pd.read_csv(path_data)

    logging.info(f"Orig Data Shape: {data.shape}")
    
    # check how much missing data in each row
    mvr_df = data.isnull().mean(axis=1)
    data_removed_df = data.loc[mvr_df[mvr_df >= 0.20].index]
    logging.info(f"Removed Samples with over 20% missing data by row: {data_removed_df.shape}")
    data_new_df = data.loc[mvr_df[mvr_df < 0.20].index]
    logging.info(f"Shape: {data_new_df.shape}")
    logging.info(f"Age 12 Depressed of data with under 20% missing by"
        f" row:\n{data_new_df[feature_label_12].value_counts()}")
    data_new_df[data[feature_label_12].notna()].to_csv(output_file_12, index=False)
    logging.info(f"Age 16 Depressed of data with under 20% missing by"
        f" row:\n{data_new_df[feature_label_16].value_counts()}")
    data_new_df[data[feature_label_16].notna()].to_csv(output_file_16, index=False)
    logging.info(f"Age 17 Depressed of data with under 20% missing by"
        f" row:\n{data_new_df[feature_label_17].value_counts()}")
    data_new_df[data[feature_label_17].notna()].to_csv(output_file_17, index=False)
    logging.info(f"Age 18 Depressed of data with under 20% missing by"
        f" row:\n{data_new_df[feature_label_18].value_counts()}")
    data_new_df[data[feature_label_18].notna()].to_csv(output_file_18, index=False)



if __name__ == '__main__':
    main()
