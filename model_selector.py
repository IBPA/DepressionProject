# -*- coding: utf-8 -*-
"""Model grid searcher module.

Authors:
    Fangzhou Li - fzli@ucdavis.edu

Todo:

"""
from typing import Tuple
import logging

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd

# from .preprocessor import Preprocessor
from .classifier_handler import ClassifierHandler
from .utils.parse_cv_results_ import parse_cv_results

# might need to change n_jobs = 1
class ModelSelector:
    """Model grid searcher class.

    It performs grid search with CV for a given classifier.

    Args:
        classifier_mode: Specification for a classifier.
            {'dummyclassifier',
             'gaussiannb',
             'multinomialnb',
             'svc',
             'adaboostclassifier',
             'decisiontreeclassifier',
             'randomforestclassifier',
             'mlpclassifier'}, default='dummyclassifier'.
        scale_mode: Specification for a scaling method.
            {'standard',
             'minmax',
             'robust'}, default='standard'.

    Attributes:
        clf: A classifier with `fit` function.

    """

    def __init__(
            self,
            classifier_mode: str = 'dummyclassifier',
            scoring='f1'):
        self.classifier_mode = classifier_mode
        self.scoring = scoring

        self.clf = ClassifierHandler(classifier_mode=self.classifier_mode).clf

    def run_model_grid_search_cv(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            param_grid: dict = None) -> Tuple[dict, float]:
        """Performs grid search.

        Args:
            X: Input data.
            y: Target data.
            param_grid: Parameters for grid search.

        Returns:
            best_params: The parameters of the best model.
            best_score: The score of the the best model

        """
        grid_search_cv = GridSearchCV(
            estimator=self.clf,
            param_grid=param_grid,
            scoring=['f1', 'precision', 'recall', 'accuracy'],
            refit='f1',
            return_train_score = True,
            n_jobs=-1,
            verbose=1)
        grid_search_cv.fit(X, y)
        best_params, best_score = parse_cv_results(
            grid_search_cv.cv_results_,
            report_score_using='f1',
            scoring=['f1', 'precision', 'recall', 'accuracy'])

        return best_params, best_score

    def run_model_cv(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            random_state: int = None) -> float:
        """Run CV without grid search for some models.

        Args:
            X: Input data.
            y: Target data.
            random_state: Random seed.

        Returns:
            score_avg: The average score of the run.

        """
        skf = StratifiedKFold(shuffle=True, random_state=random_state)
        scores = []
        train_scores = []
        y_trues = ()
        y_preds = ()
        y_probs = ()

        for idx_train, idx_test in skf.split(X, y):
            X_train = X.iloc[idx_train, :]
            X_test = X.iloc[idx_test, :]
            y_train = y.iloc[idx_train]
            y_test = y.iloc[idx_test]

            self.clf.fit(X_train, y_train)

            y_pred = self.clf.predict(X_test)
            y_train_pred = self.clf.predict(X_train)
            y_prob = pd.DataFrame(
                self.clf.predict_proba(X_test),
                columns=self.clf.classes_)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            logging.debug(
                (f"Confusion matrix (tn, fp, fn, tp): "
                 f"({tn} {fp} {fn} {tp})"))

            y_trues += (y_test,)
            y_preds += (y_pred,)
            y_probs += (y_prob,)

            if self.scoring.lower() == 'f1':
                scores.append(f1_score(y_test, y_pred))
                train_scores.append(f1_score(y_train,y_train_pred))
            elif self.scoring.lower() == 'accuracy':
                scores.append(accuracy_score(y_test, y_pred))
                train_scores.append(accuracy_score(y_train,y_train_pred))
            else:
                raise ValueError('Invalid scoring: {}'.format(self.scoring))

        logging.info(f"{self.scoring} for each fold: {scores}")
        logging.info(f"{self.scoring} for train for each fold: {train_scores}")

        score_avg = np.mean(scores)
        score_std = np.std(scores)

        train_score_avg = np.mean(train_scores)
        train_score_std = np.std(train_scores)

        logging.info(f"{self.scoring} score: {score_avg} +/- {score_std}")
        logging.info(f"{self.scoring} train score: {train_score_avg} +/- {train_score_std}")

        return score_avg
