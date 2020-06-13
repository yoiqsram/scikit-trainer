from abc import ABCMeta, abstractmethod
from functools import partial

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, is_classifier, is_regressor
from sklearn.ensemble import StackingClassifier, StackingRegressor

__all__ = ['ModelStackClassifier', 'ModelStackRegressor']


class ModelStack(TransformerMixin, BaseEstimator,
                 metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, layers: list or tuple, stack_method='auto', passthrough=False,
                 cv: int = None, n_jobs: int = None, verbose: int = 0):
        assert len(layers) > 1
        self.layers = layers

        self.stack_method = stack_method
        self.passthrough = passthrough
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _stack_layers(self, *layers, default=None):
        if is_classifier(self):
            stack = partial(StackingClassifier, cv=self.cv,
                            n_jobs=self.n_jobs, passthrough=self.passthrough, verbose=0)
        elif is_regressor(self):
            stack = partial(StackingRegressor, cv=self.cv,
                            n_jobs=self.n_jobs, passthrough=self.passthrough, verbose=0)

        layers = list(layers)

        if len(layers) == 1:
            if isinstance(layers[0], list):
                [(str(i) + '_' + layers[0][i].__class__.__name__, layers[0][i]) for i in range(len(layers[0]))]
                return stack(layers[0], default)
            else:
                return layers[0]

        elif len(layers) == 2:
            if not isinstance(layers[0], list):
                layers[0] = [layers[0]]

            if not isinstance(layers[1], list):
                layers[1] = [layers[1]]

            layers[0] = [(str(i) + '_' + layers[0][i].__class__.__name__, layers[0][i]) for i in range(len(layers[0]))]

            return self._stack_layers(*[stack(layers[0], estimator) for estimator in layers[1]], default=default)

        elif len(layers) > 2:
            return self._stack_layers(self._stack_layers(layers[0], layers[1], default=default),
                                      *layers[2:], default=default)

        raise Exception

    def fit(self, X, y, sample_weight=None):
        stacked_layers = self._stack_layers(*self.layers)
        self.estimator_ = stacked_layers.fit(X, y, sample_weight)

        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class ModelStackClassifier(ClassifierMixin, ModelStack):
    def __init__(self, layers: list or tuple, stack_method='auto', passthrough=False,
                 cv: int = None, n_jobs: int = None, verbose: int = 0):
        super().__init__(layers=layers, stack_method=stack_method, passthrough=passthrough,
                         cv=cv, n_jobs=n_jobs, verbose=verbose)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class ModelStackRegressor(RegressorMixin, ModelStack):
    def __init__(self, layers: list or tuple, stack_method='auto', passthrough=False,
                 cv: int = None, n_jobs: int = None, verbose: int = 0):
        super().__init__(layers=layers, stack_method=stack_method, passthrough=passthrough,
                         cv=cv, n_jobs=n_jobs, verbose=verbose)
