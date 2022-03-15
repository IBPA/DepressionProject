class DefaultConfig:
    @classmethod
    def get_params(cls):
        return BestAdaBoostClassifierConfig.get_params()


class BestModelConfig:
    CLASSIFIER_MODE = None
    SCALING_MODE = None
    MVI_MODE = None
    OUTLIER_MODE = None
    PARAMS = None

    @classmethod
    def get_params(cls):
        """
        """
        return {
            'classifier_mode': cls.CLASSIFIER_MODE,
            'scale_mode': cls.SCALING_MODE,
            'impute_mode': cls.MVI_MODE,
            'outlier_mode': cls.OUTLIER_MODE,
            'params': cls.PARAMS
        }


class BestDummyClassifierConfig(BestModelConfig):
    CLASSIFIER_MODE = 'dummyclassifier'
    SCALING_MODE = 'robust'
    MVI_MODE = 'missforest'
    OUTLIER_MODE = 'lof'
    PARAMS = None


class BestDecisionTreeClassifierConfig(BestModelConfig):
    CLASSIFIER_MODE = 'decisiontreeclassifier'
    SCALING_MODE = 'robust'
    MVI_MODE = 'iterative'
    OUTLIER_MODE = 'lof'
    PARAMS = {
        'criterion': 'entropy',
        'max_depth': 3,
        'min_samples_leaf': 7,
        'min_samples_split': 4,
        'random_state': 42,
        'splitter': 'random'
    }



class BestGaussianNBConfig(BestModelConfig):
    CLASSIFIER_MODE = 'gaussiannb'
    SCALING_MODE = 'standard'
    MVI_MODE = 'iterative'
    OUTLIER_MODE = 'lof'
    PARAMS = None


class BestMultinomialNBConfig(BestModelConfig):
    CLASSIFIER_MODE = 'multinomialnb'
    SCALING_MODE = 'minmax'
    MVI_MODE = 'missforest'
    OUTLIER_MODE = 'lof'
    PARAMS = None


class BestSVCConfig(BestModelConfig):
    CLASSIFIER_MODE = 'svc'
    SCALING_MODE = 'minmax'
    MVI_MODE = 'missforest'
    OUTLIER_MODE = 'lof'
    PARAMS = None


class BestAdaBoostClassifierConfig(BestModelConfig):
    CLASSIFIER_MODE = 'adaboostclassifier'
    SCALING_MODE = 'robust'
    MVI_MODE = 'iterative'
    OUTLIER_MODE = 'lof'
    PARAMS = {
        'algorithm': 'SAMME',
        'learning_rate': 0.1,
        'n_estimators': 50,
        'random_state': 42
    }


class BestRandomForestClassifierConfig(BestModelConfig):
    CLASSIFIER_MODE = 'randomforestclassifier'
    SCALING_MODE = 'robust'
    MVI_MODE = 'iterative'
    OUTLIER_MODE = 'lof'
    PARAMS = {
        'criterion': 'gini',
        'min_samples_leaf': 9,
        'min_samples_split': 4,
        'n_estimators': 75,
        'random_state': 42
    }

class BestMLPClassifierConfig(BestModelConfig):
    CLASSIFIER_MODE = 'mlpclassifier'
    SCALING_MODE = 'robust'
    MVI_MODE = 'iterative'
    OUTLIER_MODE = 'lof'
    PARAMS = {
        'hidden_layer_sizes': (20, 20, 20, 20, 20),
        'random_state': 42
    }
