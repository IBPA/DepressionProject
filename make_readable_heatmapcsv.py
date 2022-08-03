"""Run univariate feature selection and
    compare to rfe output

Authors:
    Fangzhou Li - fzli@ucdavis.edu
    Arielle Yoo - asmyoo@ucdavis.edu
"""

import os
import pickle
import logging
from statistics import stdev

import numpy as np
import pandas as pd
import click
import scipy as sp
from sklearn.metrics import precision_score
from ast import literal_eval
from functools import reduce

from .run_univariate import make_readable


@click.command()
@click.argument(
    'path-input-dir',
    type=click.Path(exists=True))
def main(
        path_input_dir):
    pearson = pd.read_csv(f"{path_input_dir}/pearson.csv")
    pearson["X_readable"] = make_readable(pearson["X"])
    pearson["Y_readable"] = make_readable(pearson["Y"])
    # print(pearson)
    pearson.to_csv(f"{path_input_dir}/pearson_readable.csv", index=False)


if __name__ == '__main__':
    main()
