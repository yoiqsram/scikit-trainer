from typing import Callable, Dict, List, Tuple, Union, Iterable

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV

from sktrainer._estimator import classifier, regressor

__all__ = ['ModelTrainer', 'train']


def _check_estimator(estimator, task):
    return isinstance(estimator, BaseEstimator) and \
           ((isinstance(estimator, RegressorMixin) and task == 'reg') or
            (isinstance(estimator, ClassifierMixin) and task == 'classif'))


class ModelTrainer(BaseEstimator):
    _estimator = {'reg': regressor, 'classif': classifier}

    _scoring = {
        'classif': ['accuracy'],
        'reg': ['neg_root_mean_squared_error']
    }

    def __init__(self, task: str = 'reg', config: Union[str, List, Tuple, Dict] = 'all',
                 scoring: str = None, cv: int = 5, n_jobs: int = -1,
                 verbose: int = 1, random_state: int = None):
        assert task in {'reg', 'classif'}
        self.task = task

        self.config = self._get_config(config)

        if isinstance(scoring, str):
            scoring = [scoring]
        elif scoring is None:
            scoring = self._scoring[self.task]

        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.summary = None

    def _get_config(self, config: Union[str, Dict, Callable]):
        if isinstance(config, type):
            config = config()

        if isinstance(config, str):
            if config == 'all':
                return {estimator(): dict() for estimator in self._estimator[self.task].values()}
            elif config in self._estimator[self.task]:
                return {self._estimator[self.task][config](): dict()}

        elif _check_estimator(config, self.task):
            return {config: dict()}

        elif isinstance(config, Dict):
            return {list(self._get_config(key).keys())[0]: value for key, value in config.items()}

        elif isinstance(config, Iterable):
            _config = dict()
            for conf in config:
                _config.update(self._get_config(conf))
            return _config

        raise ValueError

    def _fit(self, X, y, estimator, params):
        prefix = ''
        if self.task == 'classif':
            prefix = 'base_estimator__'
            estimator = CalibratedClassifierCV(base_estimator=estimator)

        search = params.pop('search', 'grid')
        if search == 'grid':
            estimator_ = GridSearchCV(estimator,
                                      {prefix + key: value for key, value in params.items()},
                                      scoring=self.scoring,
                                      refit=self.scoring[0],
                                      cv=self.cv,
                                      n_jobs=self.n_jobs).fit(X, y)
        else:
            n_iter = params.get('n_iter', 5)
            if 'n_iter' in params:
                del params['n_iter']

            estimator_ = RandomizedSearchCV(estimator,
                                            {prefix + key: value for key, value in params.items()},
                                            scoring=self.scoring,
                                            refit=self.scoring[0],
                                            cv=self.cv,
                                            n_jobs=self.n_jobs,
                                            n_iter=n_iter,
                                            random_state=self.random_state).fit(X, y)

        return estimator_

    def _record_results(self, estimator_search, estimator_name):
        self.estimators_.append(estimator_search)

        result_ = {
            key: value[i]
            for key, value in estimator_search.cv_results_.items()
            for i in range(len(estimator_search.cv_results_['params']))
            if i == estimator_search.best_index_
        }

        for key, value in result_.items():
            if key in self.results_:
                self.results_[key] = np.append(self.results_.get(key), value)
                if key[:4] == 'rank':
                    score_rank = self.results_['mean' + key[4:]]
                    self.results_[key] = len(score_rank) - np.argsort(score_rank)
            else:
                self.results_[key] = np.array([value])

        self.best_index_ = np.argmin(self.results_['rank_test_' + self.scoring[0]])
        self.best_score_ = self.results_['rank_test_' + self.scoring[0]][self.best_index_]
        self.best_estimator_ = self.estimators_[self.best_index_]

        if self.verbose:
            print('\nBest score:')
            for key, values in result_.items():
                if key[:9] == 'mean_test':
                    print('  {}: {:.4f}'.format(key[10:], values), end=' ')
                elif key[:8] == 'std_test':
                    print('+/- {:.4f}'.format(values))
            print('=' * 50, end='\n\n')

        summary_ = {'model': estimator_name, 'params': str(result_['params'])}
        for scoring in self.scoring:
            summary_['mean ' + scoring] = result_['mean_test_' + scoring]
            summary_['std ' + scoring] = result_['std_test_' + scoring]

        self.summary = self.summary.append(pd.Series(summary_), ignore_index=True)

    def fit(self, X, y):
        self.estimators_ = []
        self.results_ = dict()

        self.summary = pd.DataFrame(columns=['model', 'params'] +
                                            [f'{prefix} {scoring}' for scoring in self.scoring
                                             for prefix in ('mean', 'std')])

        for estimator, params in self.config.items():
            estimator_name = estimator.__class__.__name__

            params['search'] = params.get('search', 'grid')

            if self.verbose:
                print(estimator_name,
                      '\n'.join(f'  {key}: {str(values)}' for key, values in params.items()),
                      sep='\n')

            estimator_ = self._fit(X, y, estimator, params)
            self._record_results(estimator_, estimator_name)

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

    def get_estimator(self, key: int or str = 'best'):
        assert key in range(len(self.estimators_)) or key == 'best'

        if key == 'best':
            return self.best_estimator_
        elif isinstance(key, int):
            return self.estimators_[key]

    def predict(self, X, key: int or str = 'best'):
        return self.get_estimator(key).predict(X)

    def predict_proba(self, X, key: int or str = 'best'):
        return self.get_estimator(key).predict_proba(X)

    def predict_log_proba(self, X, key: int or str = 'best'):
        return self.get_estimator(key).predict_log_proba(X)


def train(task='reg', config='all', scoring=None, cv=5, n_jobs=None, verbose=1, random_state=None):
    return ModelTrainer(task=task,
                        config=config,
                        scoring=scoring,
                        cv=cv,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        random_state=random_state)
