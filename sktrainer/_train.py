from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier, is_regressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, check_cv
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.utils.validation import check_is_fitted, _check_fit_params
from sklearn.utils.metaestimators import if_delegate_has_method

from ._utils import validate_config, validate_estimator

__all__ = ['ModelClassifier', 'ModelRegressor']


class _Model(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config: Union[str, List, Tuple, Dict], scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=1, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=True, random_state=None):
        self.config = config
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.random_state = random_state

        self.summary = None

    def score(self, X, y=None):
        check_is_fitted(self)

        if self.scorer_ is None:
            raise ValueError

        score = self.scorer_[self.refit] if self.multimetric_ else self.scorer_
        return score(self.best_estimator_, X, y)

    @if_delegate_has_method(delegate='best_estimator_')
    def predict(self, X):
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    @if_delegate_has_method(delegate='best_estimator_')
    def predict_proba(self, X):
        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    @if_delegate_has_method(delegate='best_estimator_')
    def predict_log_proba(self, X):
        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)

    @if_delegate_has_method(delegate='best_estimator_')
    def decision_function(self, X):
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        check_is_fitted(self)
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, X):
        check_is_fitted(self)
        return self.best_estimator_.inverse_transform(X)

    @property
    def classes_(self):
        check_is_fitted(self)
        return self.best_estimator_.classes_

    def _validate_estimators(self):
        _estimators, params = validate_config(self.config)

        is_estimator_type = (is_classifier if is_classifier(self)
                             else is_regressor)

        estimators = []
        for i in range(len(_estimators)):
            estimators.append(validate_estimator(_estimators[i], self._estimator_type))

            if not is_estimator_type(estimators[i]):
                raise ValueError

        return estimators, params

    def fit(self, X, y=None, **fit_params):
        cv = check_cv(self.cv, y, classifier=is_classifier(self))
        fit_params = _check_fit_params(X, fit_params)

        scorers, self.multimetric_ = _check_multimetric_scoring(self, scoring=self.scoring)
        best_scorer = list(scorers.keys())[0]

        self.results_ = {}
        self.summary = pd.DataFrame(columns=['model', 'params'] +
                                            [f'{prefix} {scoring}' for scoring in scorers.keys()
                                             for prefix in ('mean', 'std')])

        self.estimators_ = []
        estimators, params = self._validate_estimators()
        for i in range(len(estimators)):
            prefix = ''
            _params = params[i].copy()
            _name = estimators[i].__class__.__name__

            if self.verbose:
                print(_name, ''.join(f'\n  {name}: {str(v)}' for name, v in _params.items()))

            search = _params.pop('search', 'grid')
            n_iter = _params.pop('n_iter', 5)

            if is_classifier(self):
                prefix = 'base_estimator__'
                estimators[i] = CalibratedClassifierCV(base_estimator=estimators[i])

            _prefixed_params = {prefix + name: v for name, v in _params.items()}

            if search == 'grid':
                estimator_ = GridSearchCV(estimators[i], _prefixed_params,
                                          scoring=self.scoring,
                                          refit=self.refit,
                                          cv=self.cv,
                                          n_jobs=self.n_jobs).fit(X, y)
            else:
                estimator_ = RandomizedSearchCV(estimators[i], _prefixed_params,
                                                scoring=self.scoring,
                                                refit=self.refit,
                                                cv=self.cv,
                                                n_jobs=self.n_jobs,
                                                n_iter=n_iter,
                                                random_state=self.random_state).fit(X, y)

            self.estimators_.append(estimator_)

            for name, v in estimator_.cv_results_.items():
                v = v[estimator_.best_index_]
                v = {key[len(prefix):]: value for key, value in v.items()} if name == 'params' else v

                if name in self.results_:
                    self.results_[name] = np.append(self.results_[name], v)

                    if name[:4] == 'rank':
                        score_rank = self.results_['mean' + name[4:]]
                        self.results_[name] = len(score_rank) - np.argsort(score_rank)
                else:
                    self.results_[name] = np.array([v])

            summary = {'model': _name,
                       'params': '; '.join(f'{name}: {v}' for name, v in self.results_['params'][-1].items())}
            for scorer in scorers.keys():
                summary['mean ' + scorer] = self.results_['mean_test_' + scorer][-1]
                summary['std ' + scorer] = self.results_['std_test_' + scorer][-1]

            self.summary = self.summary.append(pd.Series(summary), ignore_index=True)

            self.best_index_ = np.argmin(self.results_['rank_test_' + best_scorer])
            self.best_score_ = self.results_['rank_test_' + best_scorer][self.best_index_]
            self.best_estimator_ = self.estimators_[self.best_index_]

            if self.verbose:
                print('\nBest score:')
                for name, v in self.results_.items():
                    if name[:9] == 'mean_test':
                        print('  {}: {:.4f}'.format(name[10:], v[-1]), end=' ')
                    elif name[:8] == 'std_test':
                        print('+/- {:.4f}'.format(v[-1]))
                print('=' * 50, end='\n\n')

        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        if self.verbose:
            print('Best model:')
            print(f"  {self.summary['model'].iloc[self.best_index_]}")
            print('\n'.join(f'    {key}: {str(value)}'
                            for key, value in self.results_['params'][self.best_index_].items()))
            for key, value in self.results_.items():
                if key[:9] == 'mean_test':
                    print('  {}: {:.4f}'.format(key[10:], value[self.best_index_]), end=' ')
                elif key[:8] == 'std_test':
                    print('+/- {:.4f}'.format(value[self.best_index_]))

        return self


class ModelClassifier(ClassifierMixin, _Model):
    def __init__(self, config: Union[str, List, Tuple, Dict], scoring=None, cv=None,
                 n_jobs=None, verbose=1, pre_dispatch='2*n_jobs', random_state=None):
        super().__init__(config=config, scoring=scoring, cv=cv, n_jobs=n_jobs,
                         verbose=verbose, pre_dispatch=pre_dispatch, random_state=random_state)


class ModelRegressor(RegressorMixin, _Model):
    def __init__(self, config: Union[str, List, Tuple, Dict], scoring=None, cv=None,
                 n_jobs=None, verbose=1, pre_dispatch='2*n_jobs', random_state=None):
        super().__init__(config=config, scoring=scoring, cv=cv, n_jobs=n_jobs,
                         verbose=verbose, pre_dispatch=pre_dispatch, random_state=random_state)
