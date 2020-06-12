from typing import Callable, Dict, List, Tuple, Generator, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV

from sktrainer._estimator import classifier, regressor

__all__ = ['ModelTrainer', 'train']


_estimator = {'reg': regressor, 'classif': classifier}


def _check_estimator(estimator, task):
    return isinstance(estimator, BaseEstimator) and \
           ((isinstance(estimator, RegressorMixin) and task == 'reg') or
            (isinstance(estimator, ClassifierMixin) and task == 'classif'))


def _get_estimator(estimator, task):
    if estimator in _estimator[task]:
        return _estimator[task][estimator]()
    elif _check_estimator(estimator(), task):
        return estimator()
    elif _check_estimator(estimator, task):
        return estimator

    raise ValueError


def _transform_config(config: Union[str, dict, list, tuple, Generator, Callable], task):
    if config == 'all':
        estimators = _estimator[task].values()
        return [{'name': estimator.__class__.__name__, 'estimator': estimator()} for estimator in estimators]

    elif isinstance(config, dict):
        config['estimator'] = _get_estimator(config['estimator'], task)
        config['name'] = config.get('name', config['estimator'].__class__.__name__)
        return [config]

    elif any(isinstance(config, t) for t in (list, tuple, Generator)):
        _config = []
        for conf in config:
            _config.extend(_transform_config(conf, task))
        return _config

    else:
        estimator = _get_estimator(config, task)
        return [{'name': estimator.__class__.__name__, 'estimator': estimator}]


class ModelTrainer(BaseEstimator):
    _default_scoring = {
        'classif': ['accuracy'],
        'reg': ['neg_root_mean_squared_error']
    }

    def __init__(self, task: str = 'reg', config: Union[str, List, Tuple, Dict] = 'all',
                 scoring: str = None, cv: int = 5, n_jobs: int = -1, verbose: int = 1,
                 random_state: int = None, ):
        assert task in {'reg', 'classif'}
        self.task = task

        self.config = _transform_config(config, self.task)

        if isinstance(scoring, str):
            scoring = [scoring]
        elif scoring is None:
            scoring = self._default_scoring[self.task]

        self.scoring = scoring

        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        self.summary = None

    def _fit(self, X, y, conf):
        conf = conf.copy()
        estimator = conf.pop('estimator')
        estimator_name = conf.pop('name')
        search = conf.pop('search', 'grid')
        n_iter = conf.pop('n_iter', 5)

        if self.verbose:
            print(estimator_name, '\n'.join(f'  {name}: {str(v)}' for name, v in conf.items()), sep='\n')

        prefix = ''
        if self.task == 'classif':
            prefix = 'base_estimator__'
            estimator = CalibratedClassifierCV(base_estimator=estimator)

        params = {prefix + name: v for name, v in conf.items()}

        if search == 'grid':
            estimator_ = GridSearchCV(estimator, params,
                                      scoring=self.scoring,
                                      refit=self.scoring[0],
                                      cv=self.cv,
                                      n_jobs=self.n_jobs).fit(X, y)
        else:
            estimator_ = RandomizedSearchCV(estimator, params,
                                            scoring=self.scoring,
                                            refit=self.scoring[0],
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

        summary_ = {'model': estimator_name,
                    'params': '; '.join(f'{name}: {v}' for name, v in self.results_['params'][-1].items())}
        for scoring in self.scoring:
            summary_['mean ' + scoring] = self.results_['mean_test_' + scoring][-1]
            summary_['std ' + scoring] = self.results_['std_test_' + scoring][-1]

        self.summary = self.summary.append(pd.Series(summary_), ignore_index=True)

        self.best_index_ = np.argmin(self.results_['rank_test_' + self.scoring[0]])
        self.best_score_ = self.results_['rank_test_' + self.scoring[0]][self.best_index_]
        self.best_estimator_ = self.estimators_[self.best_index_]

        if self.verbose:
            print('\nBest score:')
            for name, v in self.results_.items():
                if name[:9] == 'mean_test':
                    print('  {}: {:.4f}'.format(name[10:], v[-1]), end=' ')
                elif name[:8] == 'std_test':
                    print('+/- {:.4f}'.format(v[-1]))
            print('=' * 50, end='\n\n')

        return estimator_

    def fit(self, X, y):
        self.estimators_ = []
        self.results_ = dict()

        self.summary = pd.DataFrame(columns=['model', 'params'] +
                                            [f'{prefix} {scoring}' for scoring in self.scoring
                                             for prefix in ('mean', 'std')])

        for conf in self.config:
            self._fit(X, y, conf)

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

        raise KeyError

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
