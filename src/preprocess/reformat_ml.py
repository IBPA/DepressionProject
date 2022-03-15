"""
"""
import argparse
import copy
from functools import partial
from multiprocessing import cpu_count, Pool
import random
import sys
from typing import List, Tuple
sys.path.append('..')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from utils import logging  # noqa: E402
from utils.utils import calc_chunksize, read_lines  # noqa: E402
logger = logging.set_logging(__name__)

DEFAULT_RAW_DATA = '../../data/Dataset012322.csv'
# DEFAULT_RAW_DATA = '../../data/small.csv'
DEFAULT_MAPPING = '../../data/Variables013122new.csv'
DEFAULT_OUTPUT_FILE_WITHOUT_TEMPORAL = '../../output/preprocessed_data_without_temporal.txt'
DEFAULT_OUTPUT_FILE_WITH_TEMPORAL = '../../output/preprocessed_data_with_temporal.txt'
DEFAULT_LOG_LEVEL = 'DEBUG'

_random = None


def parse_argument() -> argparse.Namespace:
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Preprocess raw data.')

    parser.add_argument(
        '--raw_data',
        default=DEFAULT_RAW_DATA,
        type=str,
        help='Filepath of the raw data.')

    parser.add_argument(
        '--mapping_data',
        default=DEFAULT_MAPPING,
        type=str,
        help='Filepath of the variable mapping data.')

    parser.add_argument(
        '--output_file_without_temporal',
        default=DEFAULT_OUTPUT_FILE_WITHOUT_TEMPORAL,
        type=str,
        help='Filepath for the reformatted data output without temporal info.')

    parser.add_argument(
        '--output_file_with_temporal',
        default=DEFAULT_OUTPUT_FILE_WITH_TEMPORAL,
        type=str,
        help='Filepath for the reformatted data output with temporal info.')

    parser.add_argument(
        '--use_smaller_samples',
        default=None,
        type=float,
        help='Set to an integer if you wish to work on a smaller subset of data.')

    parser.add_argument(
        '--use_smaller_features',
        default=None,
        type=str,
        help='Set to a csv filepath of a list of renamed features '
             'if you wish to work on smaller set of features.')

    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='Random seed for reproducibility.')

    parser.add_argument(
        '--num_workers',
        default=cpu_count() - 1,
        type=int,
        help='Set to an interger if you wish to work on smaller set of features.')

    parser.add_argument(
        '--log_level',
        default=DEFAULT_LOG_LEVEL,
        type=str,
        help='Set log level (DEBUG|INFO|WARNING|ERROR).')

    args = parser.parse_args()

    # Check integrity.
    if args.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
        raise ValueError(f'Invalid log level: {args.log_level}')

    return args


def get_label_without_timestamp(new_label):
    """
    Returns label without timestamp from new variable name

    Inputs:
        new_label: (String) new variable name

    Outputs:
        label: (String) label portion of new variable name
    """
    separate = new_label.split("_")
    label = "_".join(separate[:-1])
    return label


def map_variables(df_data: pd.DataFrame, df_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Fill this section up later.
    """
    #assert df_mapping.shape[0] == df_data.shape[1], \
    #    'Number of features in raw data and variable mapping file does not match!'

    df_mapping_missing = df_mapping[df_mapping['renamed_variable_name'].isnull()]
    df_mapping_exists = df_mapping[~df_mapping['renamed_variable_name'].isnull()]

    # add df_mapping_missing if exists in df_data but does not exist in df_mapping
    #logger.debug(f'Variables in raw: {set(df_data.columns)}')
    #test = set(df_mapping['original_variable_name'])
    #logger.debug(f'Variables in mapping: {test}')
    missing_in_df_data = list(set(df_data.columns) - set(df_mapping['original_variable_name']))
    logger.debug(f'Variables in raw data but not in mapping: {len(missing_in_df_data)}')
    
    original_variables_without_mapping = df_mapping_missing['original_variable_name'].tolist() + missing_in_df_data
    logger.warning(
        f'Number of variables without mapping: {len(original_variables_without_mapping)}')
    logger.debug(f'Features without variable mapping:\n{original_variables_without_mapping}')

    df_renamed = df_data.copy()
    df_renamed.drop(labels=original_variables_without_mapping, axis=1, inplace=True)
    logger.debug(f'Size of data after dropping variables without mapping: {df_renamed.shape}')

    mapper = dict(zip(
        df_mapping_exists['original_variable_name'],
        df_mapping_exists['renamed_variable_name']))
    df_renamed.rename(mapper=mapper, axis=1, inplace=True)

    return df_renamed


def sort_features_by_age(
        df_renamed: pd.DataFrame,
        before_birth_representation: str = 'g',
        before_birth_week_representation: str = 'wg',
        before_birth_month_representation: str = 'mg',
        before_birth_day_representation: str = 'dg',
        month_representation: str = 'm',
        year_representation: str = 'y',
        week_representation: str = 'w',
        day_representation: str = 'd') -> Tuple[pd.DataFrame, List[str]]:
    """
    """
    ages = list(set([x.split('_')[-1] for x in list(df_renamed)]))
    logger.debug(f'Age representations in the features: {ages}')
    ages_before_birth = [x for x in ages if x.endswith(before_birth_representation)]
    ages_after_birth = [x for x in ages if not x.endswith(before_birth_representation)]

    # before birth
    def _to_years_before_birth(age):
        if before_birth_month_representation in age:
            age_in_years = int(age.replace(before_birth_month_representation, ''))
            age_in_years /= 12
        elif before_birth_week_representation in age:
            age_in_years = int(age.replace(before_birth_week_representation, ''))
            age_in_years /= 52
        elif before_birth_day_representation in age:
            age_in_years = int(age.replace(before_birth_day_representation, ''))
            age_in_years /= 365
        else:
            raise ValueError('Invalid age format!')

        return age_in_years

    def _sort_before_birth(input_list):
        np_input = np.array(input_list)
        np_input_in_years = np.array([_to_years_before_birth(x) for x in input_list])
        index_array = np.argsort(np_input_in_years)
        return np_input[index_array]

    ages_before_birth = _sort_before_birth(ages_before_birth)

    # after birth
    def _to_years(age):
        if month_representation in age:
            age_in_years = int(age.replace(month_representation, ''))
            age_in_years /= 12
        elif year_representation in age:
            age_in_years = int(age.replace(year_representation, ''))
        elif week_representation in age:
            age_in_years = int(age.replace(week_representation, ''))
            age_in_years /= 52
        elif day_representation in age:
            age_in_years = int(age.replace(day_representation, ''))
            age_in_years /= 365
        else:
            raise ValueError('Invalid age format!')

        return age_in_years

    def _sort_after_birth(input_list):
        np_input = np.array(input_list)
        np_input_in_years = np.array([_to_years(x) for x in input_list])
        index_array = np.argsort(np_input_in_years)
        return np_input[index_array]

    ages_after_birth = _sort_after_birth(ages_after_birth)

    sorted_age_representation = [*ages_before_birth, *ages_after_birth]
    logger.debug(f'Sorted age representation: {sorted_age_representation}')

    # now sort the columns of the data using the sorted representation
    features_sorted = {x: [] for x in sorted_age_representation}
    for f in list(df_renamed):
        age = f.split('_')[-1]
        assert age in features_sorted
        features_sorted[age].append(f)

    features_sorted = [x for k, v in features_sorted.items() for x in v]

    df_renamed_and_sorted = df_renamed.copy()
    df_renamed_and_sorted = df_renamed_and_sorted[features_sorted]

    return df_renamed_and_sorted, sorted_age_representation


def _reformat_func(
        sorted_age_representation: List[str],
        features_without_timestamp_and_child_id: List[str],
        child_id_column_name: str,
        row: pd.Series) -> str:
    """
    """
    df_subset = pd.DataFrame(
        index=sorted_age_representation,
        columns=features_without_timestamp_and_child_id,
    )

    for feature, val in row.iteritems():
        if feature == child_id_column_name:
            continue

        if val != '':
            df_subset.at[feature.rsplit('_', 1)[1], feature.rsplit('_', 1)[0]] = val
    df_subset.dropna(how='all', inplace=True)

    # child ID column added to last column of timestamped data from row
    df_subset[child_id_column_name.rsplit('_', 1)[0]] = row[child_id_column_name]
    # move child ID column to first column
    df_subset = df_subset[[child_id_column_name.rsplit('_', 1)[0]] +
                          features_without_timestamp_and_child_id]

    return df_subset.to_csv(header=False)


def reformat_data(
        df_renamed_and_sorted: pd.DataFrame,
        sorted_age_representation: List[str],
        output_file: str,
        num_workers: int,
        child_id_column_name: str = 'cidB2846_0m') -> None:
    """
    """
    features_with_timestamp = list(df_renamed_and_sorted)
    logger.debug(f'Number of features with timestamp: {len(features_with_timestamp)}')

    features_without_timestamp = [x.rsplit('_', 1)[0] for x in features_with_timestamp]
    seen = set()
    seen_add = seen.add
    features_without_timestamp = [x for x in features_without_timestamp if not (x in seen or seen_add(x))]
    logger.debug(f'Number of features without timestamp: {len(features_without_timestamp)}')

    features_without_timestamp_and_child_id = copy.deepcopy(features_without_timestamp)
    features_without_timestamp_and_child_id.remove(child_id_column_name.rsplit('_', 1)[0])

    with open(output_file, mode='w', encoding='utf-8') as file:
        logger.info(f'File opened for saving reformatted data: {output_file}')
        file.write('timestamp,CH_ID,' + ','.join(features_without_timestamp_and_child_id) + '\n')

        rows = [x for _, x in df_renamed_and_sorted.iterrows()]

        chunksize = calc_chunksize(n_workers=num_workers, len_iterable=len(rows))
        logger.debug(f'Chunk size: {chunksize}')
        logger.debug(f'Number of parallel workers: {num_workers}')
        with Pool(num_workers) as p:
            for result in list(tqdm(p.imap(
                        partial(
                            _reformat_func,
                            sorted_age_representation,
                            features_without_timestamp_and_child_id,
                            child_id_column_name),
                        rows,
                        chunksize=chunksize),
                    total=len(rows))):
                file.write(result)

    logger.info(f'Reformatted data saved to \'{output_file}\'.')


def main():
    # parse arguments and set logging
    args = parse_argument()
    logger = logging.set_logging(__name__, log_level=args.log_level)
    logger.info(f'Arguments: {args}')

    # Set random seed.
    global _random
    if args.seed is not None:
        logger.info(f'Random seed set to: {args.seed}')
        _random = random.Random(args.seed)
    else:
        logger.info('Random seed is not set.')
        _random = random.Random()

    # read and process raw data
    df_raw = pd.read_csv(args.raw_data, dtype='str')
    df_raw = df_raw.applymap(lambda x: x.strip())  # remove white space
    logger.info(f'Raw data {df_raw.shape}:\n{df_raw.head()}')

    if args.use_smaller_samples:
        if args.use_smaller_samples < 1:
            df_raw = df_raw.sample(
                frac=args.use_smaller_samples,
                random_state=args.seed,
                ignore_index=True)
        else:
            df_raw = df_raw.sample(
                n=int(args.use_smaller_samples),
                random_state=args.seed,
                ignore_index=True)

        logger.info(f'Data size after sampling: {df_raw.shape}')

    # read name mapping data, csv not tab
    df_name_mapping = pd.read_csv(args.mapping_data)
    logger.info(f'Variable mapping data {df_name_mapping.shape}:\n{df_name_mapping.head()}')

    # Map variables
    logger.info('Mapping variables...')
    df_renamed = map_variables(df_raw, df_name_mapping)
    logger.info(f'Data after variable renaming:\n{df_renamed.head()}')

    if args.use_smaller_features:
        smaller_features = read_lines(args.use_smaller_features)
        logger.info(f'Number of smaller features: {len(smaller_features)}')

        invalid_smaller_features = [x for x in smaller_features if x not in df_renamed]
        valid_smaller_features = [x for x in smaller_features if x in df_renamed]

        if len(invalid_smaller_features) > 0:
            logger.warning(f'Invalid smaller features exist: {invalid_smaller_features}')
        logger.info(f'Using smaller feature subset: {valid_smaller_features}')

        df_renamed = df_renamed[valid_smaller_features]

    # Sort the age representation
    df_renamed_and_sorted, sorted_age_representation = sort_features_by_age(df_renamed)
    df_renamed_and_sorted.to_csv(args.output_file_without_temporal, index=False)
    logger.info(f'Output file without temporal data saved to: {args.output_file_without_temporal}')

    # Do final reformatting
    reformat_data(
        df_renamed_and_sorted,
        sorted_age_representation,
        args.output_file_with_temporal,
        args.num_workers)


if __name__ == '__main__':
    main()
