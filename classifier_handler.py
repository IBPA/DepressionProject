# -*- coding: utf-8 -*-
"""Classifier handler module.

Authors:
    Jason Youn - jyoun@ucdavis.edu
    Fangzhou Li - fzli@ucdavis.edu

Todo:
    - implement fit function instead of accessing attribute.

"""
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# might need to change n_jobs = 1
class ClassifierHandler:
    """Classifier handler class.

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

    Attributes:
        clf: A classifier with `fit` function.
    """

    def __init__(
            self,
            classifier_mode: str = 'dummyclassifier',
            params: dict = None,
            random_state: int = None):
        self.classifier_mode = classifier_mode
        self.params = params
        self.random_state = random_state

        self.clf = self._set_clf()

    def _set_clf(self) -> Pipeline:
        """Set `clf` attribute with a specific classifier with SMOTE sampling.

        Returns:
            clf: A sklearn classifier with `fit` function.

        Raises:
            ValueError: Invalid classifier_mode.

        """
        if self.classifier_mode == 'dummyclassifier':
            clf = DummyClassifier(
                strategy='most_frequent',
                **self.params if self.params is not None else {})
        elif self.classifier_mode == 'gaussiannb':
            clf = GaussianNB(
                **self.params if self.params is not None else {})
        elif self.classifier_mode == 'multinomialnb':
            clf = MultinomialNB(
                **self.params if self.params is not None else {})
        elif self.classifier_mode == 'svc':
            clf = SVC(
                probability=True,
                verbose=True,
                **self.params if self.params is not None else {})
        elif self.classifier_mode == 'adaboostclassifier':
            clf = AdaBoostClassifier(
                **self.params if self.params is not None else {})
        elif self.classifier_mode == 'decisiontreeclassifier':
            clf = DecisionTreeClassifier(
                **self.params if self.params is not None else {})
        elif self.classifier_mode == 'randomforestclassifier':
            clf = RandomForestClassifier(
                n_jobs=1,
                verbose=1,
                **self.params if self.params is not None else {})
        elif self.classifier_mode == 'mlpclassifier':
            clf = MLPClassifier(
                max_iter=3000,
                verbose=True,
                **self.params if self.params is not None else {})
        else:
            raise ValueError(
                f"Invalid classifier_mode: {self.classifier_mode}")

        # SMOTE up-sample
        smote = SMOTE(
            sampling_strategy='minority',
            n_jobs=1,
            random_state=self.random_state)
        clf = Pipeline([('SMOTE', smote), (self.classifier_mode, clf)])

        return clf
