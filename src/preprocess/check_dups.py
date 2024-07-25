import sys
import click
sys.path.append('..')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import logging  # noqa: E402
logger = logging.set_logging(__name__)

DEFAULT_OUTPUT_FOLDER = '../../output/'
DEFAULT_OUTPUT_FILE_WITHOUT_TEMPORAL = '../../output/preprocessed_data_without_temporal_checkdup.csv'
DEFAULT_OUTPUT_FILE_WITH_TEMPORAL = '../../output/preprocessed_data_with_temporal_checkdup.csv'
UNCLEANED_WITHINFO_FILE = '../../output/preprocessed_data_without_temporal_checkdup_withinfo.csv'
DUPS_FILE = '../../output/preprocessed_data_without_temporal_checkdup_dups.csv'
DUPS_INFO_FILE = '../../output/preprocessed_data_without_temporal_checkdup_dups_info.csv'
CLEANED_FILE = '../../output/preprocessed_data_without_temporal_checkdup_cleaned.csv'
CLEANED_INFO_FILE = '../../output/preprocessed_data_without_temporal_checkdup_cleaned_info.csv'
CLEANED_NO_INFO_FILE = '../../output/preprocessed_data_without_temporal_checkdup_cleaned_no_info.csv'
OUTPUT_FILE = 'preprocessed_data_without_temporal_checkdup_issues.csv'
INFO_COLS = ['cidB2846_0m', 'duplicated', 'duplicated_id',
             'duplicated_id_subset', 'is_empty', 'duplicated_id_subset_length']
DEPRESSION_VARS = ['y10CH_Dep_YN_127m', 'y12CH_Dep_YN_144m', 'y13CH_Dep_YN_162m', 'y16CH_Dep_YN_192m',
                   'y17CH_Dep_YN_204m', 'y18CH_Dep_YN_216m', 'y12to18_Dep_YN_216m']
ID_COL = 'cidB2846_0m'


@click.command()
@click.option('--uncleaned_withinfo_file', default=UNCLEANED_WITHINFO_FILE, help='File with uncleaned data and info')
@click.option('--dups_file', default=DUPS_FILE, help='File with duplicates')
@click.option('--dups_info_file', default=DUPS_INFO_FILE, help='File with duplicates info')
@click.option('--cleaned_file', default=CLEANED_FILE, help='File with cleaned data')
@click.option('--cleaned_info_file', default=CLEANED_INFO_FILE, help='File with cleaned data info')
@click.option('--cleaned_no_info_file', default=CLEANED_NO_INFO_FILE, help='File with cleaned data without info')
@click.option('--output_folder', default=DEFAULT_OUTPUT_FOLDER, help='Folder to save output files')
def main(uncleaned_withinfo_file: str, dups_file: str, dups_info_file: str, cleaned_file: str, cleaned_info_file: str, cleaned_no_info_file: str, output_folder: str):
    logger.info('Reading uncleaned data with info')
    df = pd.read_csv(uncleaned_withinfo_file)

    logger.info(
        'Checking duplicates - do any duplicate ids have more than one depression variable?')
    unique_ids = df[ID_COL].unique()
    # add columns to do counts for each DEPRESSION_VARS
    for var in DEPRESSION_VARS:
        df[var + '_count'] = 0
        for id_ in unique_ids:
            df.loc[df[ID_COL] == id_, var +
                   '_count'] = df.loc[df[ID_COL] == id_, var].count()
    # only save info columns + DEPRESSION_VARS columns + count columns
    cols_to_save = INFO_COLS + DEPRESSION_VARS + \
        [var + '_count' for var in DEPRESSION_VARS]
    # only save rows where any count is greater than 1
    df = df[df[[var + '_count' for var in DEPRESSION_VARS]].gt(1).any(axis=1)]
    df = df[cols_to_save]
    # move id col to end
    # df = df[[col for col in df.columns if col != ID_COL] + [ID_COL]]
    df.to_csv(output_folder + OUTPUT_FILE, index=False)


if __name__ == '__main__':
    main()
