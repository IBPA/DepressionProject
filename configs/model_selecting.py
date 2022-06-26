# -*- coding: utf-8 -*-
"""A one line summary.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * Comments

"""
import os


class DefaultConfig:
    PATH_ROOT = os.path.abspath(os.path.dirname(__file__)) + "/.."

    PATH_DATA_INPUT_FILE = (
        PATH_ROOT + "/output/preprocessed_data_without_temporal_12to18ave.csv")

    # PATH_DATA_INPUT_FILE = (
    #    PATH_ROOT + "/output/preprocessed_data_without_temporal.txt")

    # PATH_DATA_INPUT_FILE = (
    #    PATH_ROOT + "/output/has_alldep_preprocessed_data_without_temporal.csv")

    # PATH_DATA_INPUT_FILE = (
    #    PATH_ROOT + "/output/drop_samples_12_preprocessed_data_without_temporal.csv")

    # PATH_DATA_INPUT_FILE = (
    #    PATH_ROOT + "/output/drop_samples_16_preprocessed_data_without_temporal.csv")

    # PATH_DATA_INPUT_FILE = (
    #    PATH_ROOT + "/output/drop_samples_17_preprocessed_data_without_temporal.csv")

    # PATH_DATA_INPUT_FILE = (
    #    PATH_ROOT + "/output/drop_samples_18_preprocessed_data_without_temporal.csv")

    PATH_DATA_CLEANED = (
        PATH_ROOT + "/output/data_cleaned.csv")

    PATH_SFS_ANALYSIS_DIR = (
        PATH_ROOT + "/output/sfs_analysis")

    #COLUMN_DEPENDENT = 'y12CH_Dep_YN_144m'
    #COLUMN_DEPENDENT = 'y13CH_Dep_YN_162m'
    #COLUMN_DEPENDENT = 'y16CH_Dep_YN_192m'
    #COLUMN_DEPENDENT = 'y17CH_Dep_YN_204m'
    #COLUMN_DEPENDENT = 'y18CH_Dep_YN_216m'
    COLUMN_DEPENDENT = 'y12to18_Dep_YN_216m'

    # AGE_CUTOFF = 18 # year from column_dependent
    # AGE_CUTOFF = 12
    AGE_CUTOFF = 11

    @classmethod
    def get_default_preprocessed_data_path(
            cls,
            scale_mode,
            impute_mode,
            outlier_mode,
            random_state):
        """
        """
        path_output_dir = cls.PATH_ROOT + "/output"
        filename = f"data_{scale_mode}_{impute_mode}_{outlier_mode}_"\
                   f"{random_state}.pkl"

        return path_output_dir + '/' + filename

    @classmethod
    def get_default_preprocessed_sfs_prefix(
            cls,
            scale_mode,
            impute_mode,
            outlier_mode,
            random_state):
        """
        """
        path_output_dir = cls.PATH_SFS_ANALYSIS_DIR

        filename = f"sfs_{scale_mode}_{impute_mode}_{outlier_mode}_{random_state}"

        return path_output_dir + '/' + filename
