from typing import Tuple

import pandas as pd

from ..configs import ModelSelectingConfig


def load_X_and_y(
        path_data: str,
        col_dependent: str = ModelSelectingConfig.COLUMN_DEPENDENT,
        mode='pkl') -> Tuple[pd.DataFrame, pd.Series]:
    """Load X and y while dropping missing values in y.

    Args:
        path_data: The path to the input data file.
        col_dependent: The dependent variable name.
        mode: The type of data file.

    Returns:
        (X, y): Input data and target data.

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
