import copy
from functools import partial
from multiprocessing import cpu_count, Pool
import random
import sys
from typing import List, Tuple
import click
sys.path.append('..')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from utils import logging  # noqa: E402
from utils.utils import calc_chunksize, read_lines  # noqa: E402
logger = logging.set_logging(__name__)

DEFAULT_OUTPUT_FILE_WITHOUT_TEMPORAL = '../../output/preprocessed_data_without_temporal_checkdup.csv'
DEFAULT_OUTPUT_FILE_WITH_TEMPORAL = '../../output/preprocessed_data_with_temporal_checkdup.csv'
DEFAULT_OUTPUT_FOLDER = '../../output/'
DEFAULT_LOG_LEVEL = 'DEBUG'


def check_dups(df: pd.DataFrame, id_col: str = 'cidB2846_0m', temporal: bool = False) -> pd.DataFrame:
    if temporal:
        raise NotImplementedError('Not implemented yet')
    else:
        df['duplicated'] = df.duplicated(keep=False)
        df['duplicated_id'] = df.duplicated(id_col, keep=False)
        # for duplicates, is one a subset of the other?
        df['duplicated_id_subset'] = None
        for i, row in df.iterrows():
            if row['duplicated_id'] == True:
                subset = df[(df[id_col] == row[id_col])].copy(deep=True)
                # get full row data
                subset = subset.groupby(id_col).agg(
                    lambda x: list(x))
                for col in subset.columns:
                    if col not in ['duplicated', 'duplicated_id', 'duplicated_id_subset']:
                        # if all values are the same, take that value
                        if all(len(x) == 1 for x in subset[col]):
                            subset[col] = subset[col].apply(lambda x: x[0])
                        elif all(all(x[0] == y for y in x) for x in subset[col]):
                            subset[col] = subset[col].apply(lambda x: x[0])
                        else:
                            # remove all nans and nones
                            subset[col] = subset[col].apply(
                                lambda x: [y for y in x if y is not None and not pd.isna(y)])
                            # if none left, take nan or if only one value left, take that
                            subset[col] = subset[col].apply(
                                lambda x: np.nan if len(x) == 0 else x[0] if len(x) == 1 else x)
                # check if row is subset of subset when not considering created columns
                row_data = row.drop(
                    ['duplicated', 'duplicated_id', 'duplicated_id_subset'])
                subset = subset.drop(
                    ['duplicated', 'duplicated_id', 'duplicated_id_subset'], axis=1)
                # reformat subset to be same as row_data
                subset = subset.reset_index()
                print(subset)
                # save columns that are different
                diff_cols = []
                for col in subset.columns:
                    # print(
                    #     f"Subset: {subset[col][0]}, type: {type(subset[col][0])}")
                    # print(f"Row: {row_data[col]}, type: {type(row_data[col])}")
                    if not subset[col][0] == row_data[col]:
                        # only record column if missing value in row_data
                        if pd.isna(row_data[col]) and not pd.isna(subset[col][0]):
                            diff_cols.append(col)
                df.at[i, 'duplicated_id_subset'] = diff_cols
        return df


@click.command()
@click.option('--input_file', '-i', default=DEFAULT_OUTPUT_FILE_WITHOUT_TEMPORAL, help='Input file path, not temporal')
@click.option('--input_file_temporal', '-it', default=DEFAULT_OUTPUT_FILE_WITH_TEMPORAL, help='Input file path, temporal')
@click.option('--output_folder', '-o', default=DEFAULT_OUTPUT_FOLDER, help='Output folder path')
def main(input_file: str, input_file_temporal: str, output_folder: str):
    df = pd.read_csv(input_file)
    # df_temporal = pd.read_csv(input_file_temporal)

    df = check_dups(df)
    # only keep where duplicated or duplicated_id is True
    df = df[(df['duplicated'] == True) | (df['duplicated_id'] == True)]
    # df_temporal = check_dups(df_temporal, temporal=True)
    # df_temporal = df_temporal[(df_temporal['duplicated'] == True) | (df_temporal['duplicated_id'] == True)]

    df.to_csv(output_folder +
              'preprocessed_data_without_temporal_checkdup_dups.csv', index=False)
    # df_temporal.to_csv(output_folder + 'preprocessed_data_with_temporal_checkdup_dups.csv', index=False)


if __name__ == '__main__':
    main()
