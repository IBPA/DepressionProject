"""
Get the number of features and the missing value ratio of each feature in the dataset

Authors:
    Arielle Yoo - asmyoo@ucdavis.edu

"""

import pandas as pd
import os
import click

from .configs import (ModelSelectingConfig)
from .utils.visualization import plot_missing_value_ratio_histogram


def print_missing(ratio_missing, min_inclusive, min_missing, max_missing):
    if min_inclusive:
        fts_with_missing = ratio_missing[(
            ratio_missing >= min_missing) & (ratio_missing <= max_missing)]
        print(
            f"Missing [{min_missing}, {max_missing}]: {len(fts_with_missing)}")
        print(
            f"Ratio of features with missing [{min_missing}, {max_missing}]: {len(fts_with_missing) / len(ratio_missing)}"
        )
        print(
            f"Cummulative features with missing <= {max_missing}: {len(ratio_missing[ratio_missing <= max_missing]) / len(ratio_missing)}"
        )
    else:
        fts_with_missing = ratio_missing[(
            ratio_missing > min_missing) & (ratio_missing <= max_missing)]
        print(
            f"Missing ({min_missing}, {max_missing}]: {len(fts_with_missing)}")
        print(
            f"Ratio of features with missing ({min_missing}, {max_missing}]: {len(fts_with_missing) / len(ratio_missing)}"
        )
        print(
            f"Cummulative features with missing <= {max_missing}: {len(ratio_missing[ratio_missing <= max_missing]) / len(ratio_missing)}"
        )


def print_num_fts_missvalratio(path_data, path_save):
    data = pd.read_csv(path_data)
    plot_missing_value_ratio_histogram(data, path_save=path_save)
    ratio_missing = data.isnull().sum() / len(data)
    print(f"Number of features: {len(data.columns)}")
    print(f"Missing value ratio of each feature:")
    print_missing(ratio_missing, True, 0, 0.1)
    print_missing(ratio_missing, False, 0.1, 0.2)
    print_missing(ratio_missing, False, 0.2, 0.3)
    print_missing(ratio_missing, False, 0.3, 0.4)
    print_missing(ratio_missing, False, 0.4, 0.5)
    print_missing(ratio_missing, False, 0.5, 0.6)
    print_missing(ratio_missing, False, 0.6, 0.7)
    print_missing(ratio_missing, False, 0.7, 0.8)
    print_missing(ratio_missing, False, 0.8, 0.9)
    print_missing(ratio_missing, False, 0.9, 1)


@click.command()
@click.option('--path_data', default=ModelSelectingConfig.PATH_ROOT + "/output/preprocessed_data_without_temporal.txt", help='Path to the preprocessed data')
@click.option('--path_save', default=ModelSelectingConfig.PATH_ROOT + "/output/missing_value_ratio_histogram.png", help='Path to save the histogram')
def main(
    path_data: str,
    path_save: str
):
    print_num_fts_missvalratio(path_data, path_save)


if __name__ == "__main__":
    main()
