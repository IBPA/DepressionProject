# Run this after cleaning but before run_model_selection
"""Model selection running script.

Authors:
    Fangzhou Li - fzli@ucdavis.edu
    Arielle Yoo - asmyoo@ucdavis.edu

Todo:
    * Comments

"""

import click
import pandas as pd
import os
import logging

from .configs import (PreprocessingConfig, ModelSelectingConfig)
from msap.utils import (
    ClassifierHandler,
    load_X_and_y,
    dump_X_and_y,
    KFold_by_feature,
    one_hot_encode,
    binary_encode)

os.environ["PYTHONWARNINGS"] = (
    "ignore::RuntimeWarning"
)

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.DEBUG)

@click.command()
@click.argument(
    'path-input',
    type=click.Path(exists=True))
@click.argument(
    'path-output',
    type=str)
def main(
        path_input,
        path_output):
    """
    """
    # categorical columns in PreprocessingConfig.columns_categorical
    # Code from Fang's A2H code
    data = pd.read_csv(path_input)
    feature_label = ModelSelectingConfig.COLUMN_DEPENDENT
    X = data.drop([feature_label], axis=1)
    y = data[feature_label]
    vars_cat = PreprocessingConfig.columns_categorical
    cols_be = []
    cols_ohe = []
    for col in vars_cat:
        if len(X[col].value_counts() == 2):
            cols_be += [col]
        elif len(X[col].value_counts() > 2):
            cols_ohe += [col]
        else:
            raise ValueError("All NaN or constant.")
    X = binary_encode(X, cols_be)
    X = one_hot_encode(X, cols_ohe)
    X.rename(columns={'Unnamed: 0':''}, inplace=True)
    dump_X_and_y(X=X,y=y,path_output_data=path_output)

if __name__ == '__main__':
    main()
