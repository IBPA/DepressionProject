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
    ages = ['12', '12to18', '13', '16', '17', '18']
    df_all = pd.DataFrame()
    for age in ages:
        if age == '12to18':
            df = pd.read_csv(
                f"{path_input_dir}/output_{age}_yesmental/f1/feature_selection_corr.csv")
        else:
            df = pd.read_csv(
                f"{path_input_dir}/output_{age}_yesmental/feature_selection_corr.csv")
        df["codebook"] = make_readable(df["Variable"])
        df_all = pd.concat([df_all, df])
    # print(df_all)
    df_unique = df_all[["Variable", "codebook"]]
    df_unique = df_unique.drop_duplicates(subset=["Variable"])
    df_unique = df_unique.sort_values(by=["Variable"])
    df_unique.to_csv(f"{path_input_dir}/unique_features.csv", index=False)


if __name__ == '__main__':
    main()
