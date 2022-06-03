# -*- coding: utf-8 -*-
"""Model selection running script.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * include reformat.
    * I don;t like preprocessor...
    * Help for clicks

"""
import os
import pickle
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import click
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from msap.modeling.configs import (
    GridSearchConfig,
    ModelSelectionConfig)
from msap.modeling.model_selection.train import train_grid_search_cv, train_cv
from msap.modeling.model_selection.preprocessing import Preprocessor
from msap.utils import (
    ClassifierHandler,
    load_X_and_y,
    dump_X_and_y,
    KFold_by_feature)
from msap.utils.plot import (
    plot_tsne)

# need to know categorical variables
from .configs import PreprocessingConfig

os.environ["PYTHONWARNINGS"] = (
    "ignore::RuntimeWarning"
)

logger = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.DEBUG)


def preprocess(
        scale_mode: str,
        impute_mode: str,
        outlier_mode: str,
        random_state: int,
        feature_kfold,
        path_data_preprocessed_dir: str,
        X: pd.DataFrame,
        y: pd.Series,
        cat_vars: list,
        cfg_model: ModelSelectionConfig):

    filename_data_scale_impute = cfg_model.get_filename_scale_impute_data(
        scale_mode, impute_mode, outlier_mode)
    filename_data_prep = cfg_model.get_filename_preprocessed_data(
        scale_mode, impute_mode, outlier_mode)
    filename_outliers = cfg_model.get_filename_outliers(
        scale_mode, impute_mode, outlier_mode)

    # try:
    preprocessor = Preprocessor(
        scale_mode,
        impute_mode,
        outlier_mode,
        cat_vars,
        random_state,
        f"{path_data_preprocessed_dir}/"
        f"{filename_data_scale_impute}")
    X_prep, y_prep, idxs_outlier = preprocessor.preprocess(X, y)
    dump_X_and_y(
        X=X_prep
        if feature_kfold is None else X_prep.reset_index(),
        y=y_prep
        if feature_kfold is None else y_prep.reset_index(
            drop=True),
        path_output_data=f"{path_data_preprocessed_dir}/"
        f"{filename_data_prep}")
    np.savetxt(
        f"{path_data_preprocessed_dir}/{filename_outliers}",
        idxs_outlier,
        fmt='%d')
    # except Exception as e:
    #    logging.info(f"Something happened during preprocessing {e}")
    #    pass

# returns list of categorical variables from the data using
# prefixes specified in categorical


def get_all_categorical(
        categorical: list,
        X: pd.DataFrame):

    names = []
    for feature in X.columns:
        if feature.startswith(tuple(categorical)):
            names.append(feature)
    # print(names)
    indices = [X.columns.get_loc(c) for c in names if c in X]
    return indices


def change_type_to_cat(
        indices: list,
        X: pd.DataFrame):

    for i in indices:
        X.iloc[:, i] = X.iloc[:, i].astype("category")
    return X


@click.command()
@click.argument(
    'path-input',
    type=click.Path(exists=True))
@click.argument(
    'path-output',
    type=str)
@click.argument(
    'path-data-preprocessed-dir',
    type=str)
@click.argument(
    'feature-label',
    type=str)
@click.option(
    '--use-smote-first/--no-use-smote-first',
    default=False)
@click.option(
    '--feature-kfold',
    default=None)
@click.option(
    '--load-data-preprocessed/--no-load-data-preprocessed',
    default=False)
@click.option(
    '--use-categorical/--no-use-categorical',
    default=True)
@click.option(
    '--use-multiprocess-missforest/--no-use-multiprocess-missforest',
    default=True)
@click.option(
    '--random-state',
    type=int,
    default=42)
def main(
        path_input,
        path_output,
        path_data_preprocessed_dir,
        feature_label,
        use_smote_first,
        feature_kfold,
        load_data_preprocessed,
        use_categorical,
        use_multiprocess_missforest,
        random_state):
    """
    """

    cfg_model = ModelSelectionConfig

    if load_data_preprocessed is True:
        logging.info(
            "Loading preprocessed data at "
            f"{path_data_preprocessed_dir}")
    else:
        if path_data_preprocessed_dir is None:
            path_data_preprocessed_dir \
                = cfg_model.get_default_path_data_preprocessed_dir()

        logging.info(
            "Generating preprocessed data at "
            f"{path_data_preprocessed_dir}")
        if not os.path.exists(path_data_preprocessed_dir):
            os.mkdir(path_data_preprocessed_dir)

        data = pd.read_csv(path_input)
        if feature_kfold is not None:
            data = data.set_index(feature_kfold)

        X = data.drop([feature_label], axis=1)
        y = data[feature_label].astype("category")

        # test - TODO remove
        # X = X[:60]
        # y = y[:60]

        preprocess_inputs = []
        missforest_preprocess = []

        if use_categorical:
            # raise NotImplementedError
            # get full list of categorical variables
            vars_cat_prefs = PreprocessingConfig.columns_categorical
            cat_indices = get_all_categorical(vars_cat_prefs, X)
            X = change_type_to_cat(cat_indices, X)
            if len(cat_indices) == 0:
                cat_indices = None
        else:
            cat_indices = None

        for scale_mode, impute_mode, outlier_mode \
                in cfg_model.get_all_preprocessing_combinations():
            if impute_mode == 'missforest':
                # print(cat_indices)
                missforest_preprocess += [[
                    scale_mode,
                    impute_mode,
                    outlier_mode,
                    random_state,
                    feature_kfold,
                    path_data_preprocessed_dir,
                    X,
                    y,
                    cat_indices,
                    cfg_model]]
            else:
                preprocess_inputs += [[
                    scale_mode,
                    impute_mode,
                    outlier_mode,
                    random_state,
                    feature_kfold,
                    path_data_preprocessed_dir,
                    X,
                    y,
                    cat_indices,
                    cfg_model]]

        # print(len(preprocess_inputs))
        # Preprocess using multiprocessing for missforest
        logging.info("Starting Preprocessing MissForest")
        if use_multiprocess_missforest:
            with Pool() as p:  # tqdm might not be working
                p.starmap(preprocess, tqdm(missforest_preprocess,
                          total=len(missforest_preprocess)))
        else:
            for scale_mode, impute_mode, outlier_mode, random_state, \
                    feature_kfold, path_data_preprocessed_dir, X, y, \
                    cat_indices, cfg_model in tqdm(missforest_preprocess):

                preprocess(scale_mode,
                           impute_mode,
                           outlier_mode,
                           random_state,
                           feature_kfold,
                           path_data_preprocessed_dir,
                           X,
                           y,
                           cat_indices,
                           cfg_model)
        logging.info("Done Preprocessing MissForest")

        logging.info("Starting Preprocessing Not MissForest")
        # Preprocess the rest normally
        for scale_mode, impute_mode, outlier_mode, random_state, \
                feature_kfold, path_data_preprocessed_dir, X, y, \
                cat_indices, cfg_model in tqdm(preprocess_inputs):

            preprocess(scale_mode,
                       impute_mode,
                       outlier_mode,
                       random_state,
                       feature_kfold,
                       path_data_preprocessed_dir,
                       X,
                       y,
                       cat_indices,
                       cfg_model)

        # test - TODO remove return bc testing just preprocess
        # return

            # filename_data_scale_impute = cfg_model.get_filename_scale_impute_data(
            #     scale_mode, impute_mode, outlier_mode)
            # filename_data_prep = cfg_model.get_filename_preprocessed_data(
            #     scale_mode, impute_mode, outlier_mode)
            # filename_outliers = cfg_model.get_filename_outliers(
            #     scale_mode, impute_mode, outlier_mode)

            # try:
            #     preprocessor = Preprocessor(
            #         scale_mode,
            #         impute_mode,
            #         outlier_mode,
            #         random_state,
            #         f"{path_data_preprocessed_dir}/"
            #         f"{filename_data_scale_impute}")
            #     X_prep, y_prep, idxs_outlier = preprocessor.preprocess(X, y)
            #     dump_X_and_y(
            #         X=X_prep
            #         if feature_kfold is None else X_prep.reset_index(),
            #         y=y_prep
            #         if feature_kfold is None else y_prep.reset_index(
            #             drop=True),
            #         path_output_data=f"{path_data_preprocessed_dir}/"
            #         f"{filename_data_prep}")
            #     np.savetxt(
            #         f"{path_data_preprocessed_dir}/{filename_outliers}",
            #         idxs_outlier,
            #         fmt='%d')
            # except Exception as e:
            #     logging.info(f"Something happened during preprocessing {e}")
            #     pass

            # test - TODO remove
            # break

    n_total_combinations \
        = len(cfg_model.get_all_preprocessing_combinations()) \
        * len(cfg_model.get_all_classifier_modes())
    logging.info(
        "Starting the model selection pipeline for "
        f"{n_total_combinations} combinations.")

    # Iterate all combinations.
    results = []  # Store all the scores of models.
    failures = []  # Store all combinations of failed models.
    for i, (scale_mode, impute_mode, outlier_mode) in \
            enumerate(tqdm(
                cfg_model.get_all_preprocessing_combinations(),
                desc="Preprocessing Combinations")):

        filename_data_prep = cfg_model.get_filename_preprocessed_data(
            scale_mode, impute_mode, outlier_mode)

        try:
            X, y = load_X_and_y(
                f"{path_data_preprocessed_dir}/{filename_data_prep}",
                col_y=feature_label)
        except Exception as e:
            logging.debug(
                "This preprocessing, "
                f"{(scale_mode, impute_mode, outlier_mode)}, "
                "does not exist for this run.")

            for j, classifier_mode in enumerate(tqdm(
                    cfg_model.get_all_classifier_modes(),
                    desc="Classifiers")):
                failures += [
                    (i * len(cfg_model.get_all_classifier_modes()) + j,
                     (scale_mode, impute_mode, outlier_mode, classifier_mode),
                     e)]
            continue

        # Test set hold out and save to output file for loading for analysis
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state)

        train_filename = cfg_model.get_filename_preprocessed_train_data(
            scale_mode, impute_mode, outlier_mode)
        dump_X_and_y(X=X_train, y=y_train,
                     path_output_data=f"{path_data_preprocessed_dir}/{train_filename}")

        test_filename = cfg_model.get_filename_preprocessed_test_data(
            scale_mode, impute_mode, outlier_mode)
        dump_X_and_y(X=X_test, y=y_test,
                     path_output_data=f"{path_data_preprocessed_dir}/{test_filename}")

        if use_smote_first:
            # TODO - toggle for classifier smote options?
            smote = SMOTE(
                sampling_strategy='minority',
                n_jobs=1,
                random_state=cfg_model.RNG_SMOTE)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            # dump smote data to filename
            smote_train_filename = cfg_model.get_filename_smote_train_data(
                scale_mode, impute_mode, outlier_mode)
            dump_X_and_y(X=X_train_smote, y=y_train_smote,
                         path_output_data=f"{path_data_preprocessed_dir}/{smote_train_filename}")
            # Create KFold based on the specified index. Use default row id if
            #   None.
            splits = KFold_by_feature(
                X_train_smote, 5, feature_kfold, random_state)
            if feature_kfold is not None:
                X_train_smote = X_train_smote.drop([feature_kfold], axis=1)
                X_test = X_test.drop([feature_kfold], axis=1)

            for j, classifier_mode in enumerate(tqdm(
                    cfg_model.get_all_classifier_modes(),
                    desc="Classifiers")):

                clf = ClassifierHandler(
                    classifier_mode, use_smote=False).clf

                try:
                    # TODO - include test results somehow?
                    # Perform grid search and 5-fold CV if hyperparamer tuning is
                    #   available.
                    if classifier_mode in GridSearchConfig.CLASSIFIER_MODES:
                        result = train_grid_search_cv(
                            clf=clf,
                            X=X_train_smote,
                            y=y_train_smote,
                            param_grid=GridSearchConfig.get_config(
                                classifier_mode).get_param_grid(random_state),
                            splits=splits)
                    # Perform only 5-fold CV if hyperparamer tuning is not
                    #   available.
                    else:
                        result = train_cv(
                            clf=clf,
                            X=X_train_smote,
                            y=y_train_smote,
                            splits=splits)
                    results += [
                        (i * len(cfg_model.get_all_classifier_modes()) + j,
                            (scale_mode, impute_mode,
                             outlier_mode, classifier_mode),
                            result)]

                except Exception as e:
                    failures += [
                        (i * len(cfg_model.get_all_classifier_modes()) + j,
                            (scale_mode, impute_mode,
                             outlier_mode, classifier_mode),
                            e)]

            # raise NotImplementedError
        else:
            # Create KFold based on the specified index. Use default row id if
            #   None.
            splits = KFold_by_feature(X_train, 5, feature_kfold, random_state)
            if feature_kfold is not None:
                X_train = X_train.drop([feature_kfold], axis=1)
                X_test = X_test.drop([feature_kfold], axis=1)

            for j, classifier_mode in enumerate(tqdm(
                    cfg_model.get_all_classifier_modes(),
                    desc="Classifiers")):

                clf = ClassifierHandler(
                    classifier_mode, random_state=cfg_model.RNG_SMOTE).clf
                try:
                    # TODO - include test results somehow?
                    # Perform grid search and 5-fold CV if hyperparamer tuning is
                    #   available.
                    if classifier_mode in GridSearchConfig.CLASSIFIER_MODES:
                        result = train_grid_search_cv(
                            clf=clf,
                            X=X_train,
                            y=y_train,
                            param_grid=GridSearchConfig.get_config(
                                classifier_mode).get_param_grid(random_state),
                            splits=splits)
                    # Perform only 5-fold CV if hyperparamer tuning is not
                    #   available.
                    else:
                        result = train_cv(
                            clf=clf,
                            X=X_train,
                            y=y_train,
                            splits=splits)

                    results += [
                        (i * len(cfg_model.get_all_classifier_modes()) + j,
                         (scale_mode, impute_mode, outlier_mode, classifier_mode),
                         result)]

                except Exception as e:
                    failures += [
                        (i * len(cfg_model.get_all_classifier_modes()) + j,
                         (scale_mode, impute_mode, outlier_mode, classifier_mode),
                         e)]

        # test - TODO remove
        # break

    with open(path_output, 'wb') as f:
        pickle.dump((results, failures), f)


if __name__ == '__main__':
    main()
