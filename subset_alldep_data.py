# -*- coding: utf-8 -*-
"""Get data that has all depression variables

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * 

python -u -m DepressionProjectNew.subset_alldep_data \
./DepressionProjectNew/output/preprocessed_data_without_temporal.txt \
y12CH_Dep_YN_144m \
y16CH_Dep_YN_192m \
y17CH_Dep_YN_204m \
y18CH_Dep_YN_216m \
./DepressionProjectNew/output/has_alldep_preprocessed_data_without_temporal.csv
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
    'output-file',
    type=str)
def main(
        path_data,
        feature_label_12,
        feature_label_16,
        feature_label_17,
        feature_label_18,
        output_file):

    # load data that has been preprocessed but not cleaned
    data = pd.read_csv(path_data)

    logging.info(f"Orig Data Shape: {data.shape}")

    # compare data
    has_dep = data[data[feature_label_12].notna() & \
        data[feature_label_16].notna() & \
        data[feature_label_17].notna() & \
        data[feature_label_18].notna()]
    
    logging.info(f"Has All Dep Shape: {has_dep.shape}")
    has_dep.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
