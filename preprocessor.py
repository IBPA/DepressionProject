# -*- coding: utf-8 -*-
"""Data preprocessing object.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    * Should I skip scaling OHE variables?
    * Should I keep the column order after OHE?
    * Remove constant columns, where?
    * Remove overly missing columns, where?
    * docstring

"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import logging as log
import pickle
import datetime

from .impute import knn_impute, iterative_impute, missforest
from .detect_outliers import isolation_forest, local_outlier_factor
from .utils.visualization import (plot_sfs, plot_pca, plot_tsne)
# unsure if do tsne and pca plots here or in analysis

class Preprocessor:
    """Class of data preprocessing object. TODO

    Args:
        clf: A classifier with `fit` method. Optional if `skip_fs` is True.
        scale_mode: Specification for a scaling method.
            {'standard',
             'minmax',
             'robust'}, default='standard'.
        impute_mode: Specification for a missing value imputation method.
            {'knn',
             'iterative',
             'missforest'}, default='knn'.
        outlier_mode: Specification for an outlier detection method.
            {'isolation_forest',
             'lof'}, default='isolation_forest'.
        skip_fs: Skip feature selection if True, default=True.

    Raises:
        ValueError: TODO
    """

    def __init__(
            self,
            clf=None,
            scale_mode='standard',
            impute_mode='knn',
            outlier_mode='isolation_forest',
            skip_fs=True,
            visualization_prefix=None):
        self.clf = clf
        self.scale_mode = scale_mode
        self.impute_mode = impute_mode
        self.outlier_mode = outlier_mode
        self.skip_fs = skip_fs
        self.visualization_prefix = visualization_prefix

        if not self.skip_fs and self.clf is None:
            raise ValueError(
                (f"Invalid clf argument: {self.clf}."
                 " clf must exist if skip_fs is False."))

    def one_hot_encode_categorical(self, X_df, var_cat=None):
        """Encode categorical features with one-hot encoding.

        Args:
            X_df (pd.DataFrame): Input data.
            vat_cat (list of str or None): Categorical feature names. None if
                no categorical features.

        Returns:
            (pd.DataFrame): Encoded data.

        """
        if var_cat is None:
            return X_df

        def one_hot_encode_categorical_column(col):
            col_ohe = pd.get_dummies(col, prefix=col.name, dummy_na=True)
            col_ohe.loc[col_ohe[col.name + '_nan'] == 1, col_ohe.columns[:-1]]\
                = np.nan
            del col_ohe[col.name + '_nan']
            return col_ohe
        
        # Used code from Fang's A2H to fix encoding for binary encoding
        def binary_encode_column(col):
            mapped = col.value_counts().index.tolist()
            col_be = col.map({mapped[0]: 1, mapped[1]: 0})
            col_be.name = col_be.name + f'_{mapped[0]}_1_{mapped[1]}_0'
            return col_be

        cols_be = []
        cols_ohe = []
        for col in var_cat:
            if len(X_df[col].value_counts() == 2):
                cols_be += [col]
            elif len(X_df[col].value_counts() > 2):
                cols_ohe += [col]
            else:
                raise ValueError("All NaN or constant.")
        
        X_cat_encoded_lst = []
        X_df[cols_be].apply(
            lambda col: X_cat_encoded_lst.append(
                binary_encode_column(col)),
            axis=0)
        
        X_df[cols_ohe].apply(
            lambda col: X_cat_encoded_lst.append(
                one_hot_encode_categorical_column(col)),
            axis=0)

        #X_df[var_cat].apply(
        #    lambda col: X_cat_encoded_lst.append(
        #        one_hot_encode_categorical_column(col)),
        #    axis=0)

        return pd.concat(
            [X_df.drop(
                var_cat,
                axis=1)] + X_cat_encoded_lst,
            axis=1)

    def scale_features(self, X_df):
        """Scale features.

        Args:
            X_df (pd.DataFrame): Input data.

        Returns:
            (pd.DataFrame): Scaled data.

        """
        X_array = X_df.to_numpy()
        if self.scale_mode == 'standard':
            scaler = StandardScaler().fit(X_array)
        elif self.scale_mode == 'minmax':
            scaler = MinMaxScaler().fit(X_array)
        elif self.scale_mode == 'robust':
            scaler = RobustScaler().fit(X_array)
        else:
            raise ValueError(f"Invalid scaling mode: {self.scale_mode}")

        return pd.DataFrame(
            scaler.transform(X_array),
            index=X_df.index,
            columns=X_df.columns)

    def impute_missing_values(self, X_df, random_state=None):
        """Impute missing values.

        Args:
            X_df (pd.DataFrame): Input data.
            random_state (int or None): A specific random seed.

        Returns:
            (pd.DataFrame): Imputed data.

        """
        if self.impute_mode == 'knn':
            X_new_df = knn_impute(X_df)
        elif self.impute_mode == 'iterative':
            X_new_df = iterative_impute(X_df, random_state)
        elif self.impute_mode == 'missforest':
            X_new_df = missforest(X_df, random_state)
        else:
            raise ValueError(f"Invalid imputation mode: {self.impute_mode}")

        return X_new_df

    def remove_outliers(self, X_df, y_se, random_state=None):
        """Remove outliers.

        Args:
            X_df (pd.DataFrame): Input data.
            y_se (pd.Series): Target data.
            random_state (int or None): A specific random seed.

        Returns:
            (pd.DataFrame): Updated data.

        """
        if self.outlier_mode == 'isolation_forest':
            indices_bool = isolation_forest(X_df, random_state)
        elif self.outlier_mode == 'lof':
            indices_bool = local_outlier_factor(X_df)
        else:
            raise ValueError(
                f"Invalid outlier detection mode: {self.outlier_mode}")

        return X_df[indices_bool], y_se[indices_bool]

    def analyze_feature_selection(self, sfs, random_seed=None):
        """
        Analyze sequential feature selection results.
        Inputs:
            sfs: (SequentialFeatureSelector) Fitted object.
        """
        metric_dict = sfs.get_metric_dict()

        pd_metric = pd.DataFrame.from_dict(metric_dict).T
        pd_metric = pd_metric.sort_index()

        log.debug('Feature selection metric: %s', pd_metric)
        log.info('Selected features: %s', sfs.k_feature_names_)

        rank = []
        feature_names = pd_metric['feature_names'].tolist()
        for i in range(len(feature_names)-1):
            if i == 0:
                rank.append(feature_names[i][0])

            difference = list(set(feature_names[i+1]) - set(feature_names[i]))
            rank.append(difference[0])

        log.info('Feature rank from high to low: %s', rank)
        log.info(f"Visualization Prefix: {self.visualization_prefix}")
        if self.visualization_prefix:
            if random_seed is None:
                log.info("Trying to write sfs file")
                ct = datetime.datetime.now()
                plot_sfs(
                    metric_dict,
                    rank,
                    title='Sequential Backward Selection (w. StdDev)',
                    path_save=self.visualization_prefix + str(ct) + '.png')
                pickle.dump(
                    metric_dict,
                    open((self.visualization_prefix + '_metric_dict' + str(ct) + '.pkl'),'wb'))
                pickle.dump(
                    rank,
                    open((self.visualization_prefix + '_rank' + str(ct) + '.pkl'),'wb'))
            else:
                log.info("Trying to write sfs file")
                ct = datetime.datetime.now()
                plot_sfs(
                    metric_dict,
                    rank,
                    title='Sequential Backward Selection (w. StdDev)',
                    path_save=self.visualization_prefix + "_rfeseed_"
                    + str(random_seed) + "_" + str(ct) + '.png')
                pickle.dump(
                    metric_dict,
                    open((self.visualization_prefix + '_metric_dict_rfeseed_'
                        + str(random_seed) + "_" + str(ct) + '.pkl'),'wb'))
                pickle.dump(
                    rank,
                    open((self.visualization_prefix + '_rank_rfeseed_'
                        + str(random_seed) + "_" + str(ct) + '.pkl'),'wb'))
        return rank

    def select_features(self, X_df, y_se, scoring='f1', random_seed=None):
        # TODO - test
        """Operate feature selection via RFE.

        Args:
            X_df: Input data.
            y_se: Target data.
            scoring: A selection metric. 'f1' | 'accuracy'

        Returns:
            X_new: New data after feature selection.

        """
        limit = 500 #TODO - what is feasible?
        log.info(f"X_df before feature selection: {X_df.shape}")
        if len(X_df.columns) < limit:
            if random_seed is not None:
                cv = StratifiedKFold(
                        n_splits=5,
                        random_state=random_seed,
                        shuffle=True)
            else:
                cv = StratifiedKFold(n_splits=5)
            log.info("Performing feature selection")
            sfs = SFS(
                estimator=self.clf,
                k_features='parsimonious',
                forward=False,
                floating=False,
                scoring=scoring,
                cv=cv,
                n_jobs=-1,
                verbose=2)
            sfs.fit(X_df, y_se)
            rank = self.analyze_feature_selection(sfs, random_seed)

            X_new = X_df[list(sfs.k_feature_names_)]
        
        else:
            log.info("Performing Lasso for feature selection")
            # same as default, may need to adjust
            if random_seed is None:
                lasso = Lasso(alpha = 1.0)
            else:
                lasso = Lasso(alpha = 1.0, random_state=random_seed)
            model = SelectFromModel(lasso, prefit = True)
            X_new = model.transform(X_df)

        log.info(f"X_df after feature selection: {X_new.shape}")
        return X_new

    def preprocess(self, X_df, y_se, var_cat=None, random_state=None):
        """Preprocess input data.

        Args:
            X_df (pd.DataFrame): Input data.
            y_se (pd.Series): Target data.
            random_state (int or None): A specific random seed.

        Returns:
            (pd.DataFrame): Preprocessed data.

        """
        log.info("Preprocessing: Starting one hot encode")
        X_df = self.one_hot_encode_categorical(X_df, var_cat)
        log.info("Preprocessing: Starting scale features")
        X_df = self.scale_features(X_df)
        log.info("Preprocessing: Starting impute missing values")
        X_df = self.impute_missing_values(X_df, random_state)
        log.info("Preprocessing: Starting remove outliers")
        X_df, y_se = self.remove_outliers(X_df, y_se, random_state)

        if not self.skip_fs:
            X_df = self.select_features(X_df, y_se, random_seed=random_state)
            log.info("Selected Features")

        return X_df, y_se
