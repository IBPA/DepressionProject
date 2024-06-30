"""
Get the number of features and the missing value ratio of each feature in the dataset

Authors:
    Arielle Yoo - asmyoo@ucdavis.edu

"""

import pandas as pd
import os

from .configs import (ModelSelectingConfig)


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


def print_num_fts_missvalratio(path_data):
    data = pd.read_csv(path_data)
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


def main():
    path_data = ModelSelectingConfig.PATH_ROOT + \
        "/output/preprocessed_data_without_temporal.txt"
    print_num_fts_missvalratio(path_data)


if __name__ == "__main__":
    main()
