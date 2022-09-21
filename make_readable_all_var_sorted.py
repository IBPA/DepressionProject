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
    targets = ['y12CH_Dep_YN_144m', 'y12to18_Dep_YN_216m', 'y13CH_Dep_YN_162m',
               'y16CH_Dep_YN_192m', 'y17CH_Dep_YN_204m', 'y18CH_Dep_YN_216m']
    for i, age in enumerate(ages):
        df = pd.read_csv(
            f"{path_input_dir}/output_{age}_yesmental/vars_sorted_dir_ranked_rounded.csv")
        df.loc[:, "VariableName"] = df.loc[:,
                                           "VariableName"].replace("label", targets[i])
        df.loc[:, "description"] = make_readable(df.loc[:, "VariableName"])
        df.to_csv(
            f"{path_input_dir}/output_{age}_yesmental/vars_sorted_dir_ranked_rounded_readable.csv", index=False)


if __name__ == '__main__':
    main()
