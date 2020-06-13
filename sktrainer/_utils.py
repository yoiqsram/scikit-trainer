from sklearn.base import is_classifier, is_regressor

from ._estimator import classifier, regressor

__all__ = ['validate_config', 'validate_estimator']


def validate_config(config):
    if config is None or len(config) == 0:
        raise ValueError(
            "Invalid 'config', 'config' should be a list"
            " of (estimator, params) tuples."
        )

    estimators, params = zip(*config)
    return estimators, params


def validate_estimator(estimator, estimator_type=None):
    assert estimator_type in {'classifier', 'regressor', None}
    _estimator = classifier if estimator_type == 'classifier' else regressor

    if estimator in _estimator:
        return _estimator[estimator]()

    elif isinstance(estimator, type):
        return estimator()

    elif is_classifier(estimator) or is_regressor(estimator):
        return estimator

    raise ValueError
