# -*- coding: utf-8 -*-
"""Data cleaning module.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:

"""
import pandas as pd
import numpy as np
import logging

class Cleaner:
    """Class of data clearer object.

    Args:
        cols_ignored (list of str): Variable names for ignoring.
        thres_mis (float): The threshold percentage of missing values.
            Variables equal to or greater than the threshold are removed.
            Default removes variables with all missing values.

    """

    def __init__(self, cols_ignored=[], thres_mis=1.0, file_for_removed=None):
        self.cols_ignored = cols_ignored
        self.thres_mis = thres_mis
        self.file_for_removed = file_for_removed

    def remove_ignored_variables(self, data_df):
        """Remove specified ignored variables.

        Args:
            data_df (pd.DataFrame): Input data.

        Returns:
            (pd.DataFrame): Cleaned data.

        """
        data_new_df = data_df.drop(self.cols_ignored, axis=1)
        return data_new_df

    def remove_constant_variables(self, data_df):
        """Remove constant variables.

        Args:
            data_df (pd.DataFrame): Input data.

        Returns:
            (pd.DataFrame): Cleaned data.

        """
        data_new_df = data_df.loc[:, (data_df != data_df.iloc[0]).any()]
        return data_new_df

    def remove_overly_missing_variables(self, data_df):
        """Remove variables that are heavily missing.

        Args:
            data_df (pd.DataFrame): Input data.

        Returns:
            (pd.DataFrame): Cleaned data.

        """
        mvr_df = data_df.isnull().mean()
        data_removed_df = data_df[mvr_df[mvr_df >= self.thres_mis].index]
        if self.file_for_removed is not None:
            data_removed_df.to_csv(self.file_for_removed)
        data_new_df = data_df[mvr_df[mvr_df < self.thres_mis].index]
        return data_new_df

    

    def remove_later_variables(self, data_df, age_cutoff = 18):
        """Remove variables that are later or at the same time as output
        variable

        Args:
            data_df (pd.DataFrame): Input data.
            age_cutoff (int): age cutoff for data in years (data will have info
            up to but excluding that age)

        Returns:
            (pd.DataFrame): Cleaned data.

        """
        valid_age_cutoffs = [11, 12, 13, 16, 17, 18]
        assert age_cutoff in valid_age_cutoffs

        # softcoded
        def sort_features_by_age_and_drop_less_than_or_equal_to_age(
            df_renamed: pd.DataFrame,
            age_cutoff: int,
            before_birth_representation: str = 'g',
            before_birth_week_representation: str = 'wg',
            before_birth_month_representation: str = 'mg',
            before_birth_day_representation: str = 'dg',
            month_representation: str = 'm',
            year_representation: str = 'y',
            week_representation: str = 'w',
            day_representation: str = 'd') -> pd.DataFrame:
            """
            """
            ages = list(set([x.split('_')[-1] for x in list(df_renamed)]))
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

            def _remove_older_entries(age_sorted, age_cutoff):
                removed_older = np.array([x for x in age_sorted 
                                          if _to_years(x) < age_cutoff])
                return removed_older

            def _sort_after_birth_and_drop_older(input_list, age_cutoff):
                np_input = np.array(input_list)
                np_input_in_years = np.array([_to_years(x)
                                              for x in input_list])
                index_array = np.argsort(np_input_in_years)
                age_sorted = np_input[index_array]
                return _remove_older_entries(age_sorted, age_cutoff)

            ages_after_birth = _sort_after_birth_and_drop_older(
                ages_after_birth, age_cutoff)

            sorted_age_representation = [*ages_before_birth, *ages_after_birth]

            # now sort the columns of the data using the sorted representation
            features_sorted = {x: [] for x in sorted_age_representation}
            for f in list(df_renamed):
                age = f.split('_')[-1]
                # do not append if age not in features_sorted
                if age in features_sorted:
                    features_sorted[age].append(f)

            features_sorted = [x for k, v in features_sorted.items()
                               for x in v]

            df_renamed_and_sorted = df_renamed.copy()
            df_renamed_and_sorted = df_renamed_and_sorted[features_sorted]

            return df_renamed_and_sorted

        data_new_df = sort_features_by_age_and_drop_less_than_or_equal_to_age(
            data_df, age_cutoff)
        return data_new_df

    def remove_later_variables_exclusive(self, data_df, age_cutoff = 18):
        """Remove variables that are later or at the same time as output
        variable

        Args:
            data_df (pd.DataFrame): Input data.
            age_cutoff (int): age cutoff for data in years (data will have info
            up to and including that age)

        Returns:
            (pd.DataFrame): Cleaned data.

        """

        # softcoded
        def sort_features_by_age_and_drop_less_than_age(
            df_renamed: pd.DataFrame,
            age_cutoff: int,
            before_birth_representation: str = 'g',
            before_birth_week_representation: str = 'wg',
            before_birth_month_representation: str = 'mg',
            before_birth_day_representation: str = 'dg',
            month_representation: str = 'm',
            year_representation: str = 'y',
            week_representation: str = 'w',
            day_representation: str = 'd') -> pd.DataFrame:
            """
            """
            ages = list(set([x.split('_')[-1] for x in list(df_renamed)]))
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

            def _remove_older_entries(age_sorted, age_cutoff):
                removed_older = np.array([x for x in age_sorted 
                                          if _to_years(x) <= age_cutoff])
                return removed_older

            def _sort_after_birth_and_drop_older(input_list, age_cutoff):
                np_input = np.array(input_list)
                np_input_in_years = np.array([_to_years(x)
                                              for x in input_list])
                index_array = np.argsort(np_input_in_years)
                age_sorted = np_input[index_array]
                return _remove_older_entries(age_sorted, age_cutoff)

            ages_after_birth = _sort_after_birth_and_drop_older(
                ages_after_birth, age_cutoff)

            sorted_age_representation = [*ages_before_birth, *ages_after_birth]

            # now sort the columns of the data using the sorted representation
            features_sorted = {x: [] for x in sorted_age_representation}
            for f in list(df_renamed):
                age = f.split('_')[-1]
                # do not append if age not in features_sorted
                if age in features_sorted:
                    features_sorted[age].append(f)

            features_sorted = [x for k, v in features_sorted.items()
                               for x in v]

            df_renamed_and_sorted = df_renamed.copy()
            df_renamed_and_sorted = df_renamed_and_sorted[features_sorted]

            return df_renamed_and_sorted

        data_new_df = sort_features_by_age_and_drop_less_than_age(
            data_df, age_cutoff)
        return data_new_df

    def clean(self, data_df, age_cutoff = 18):
        """Clean input data by removing ignored variables, heavily missing
        variables, and constant variables.

        Args:
            data_df (pd.DataFrame): Input data.
            age_cutoff (int): age cutoff for data in years (data will have info
            up to but excluding that age)

        Returns:
            (pd.DataFrame): Cleaned data.

        """
        logging.info(f"Starting training data shape: {data_df.shape}")
        data_new_df = self.remove_ignored_variables(data_df)
        logging.info(f"Remove ignored variables shape: {data_new_df.shape}")
        data_new_df = self.remove_overly_missing_variables(data_new_df)
        logging.info(f"Remove overly missing variables shape: {data_new_df.shape}")
        data_new_df = self.remove_constant_variables(data_new_df)
        logging.info(f"Remove constant shape: {data_new_df.shape}")
        data_new_df = self.remove_later_variables(data_new_df, age_cutoff)
        logging.info(f"Remove later variables shape: {data_new_df.shape}")
        #data_new_df = self.remove_later_variables_exclusive(data_new_df, age_cutoff)
        return data_new_df
