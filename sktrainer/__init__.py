from sktrainer._train import *
from sktrainer._stack import *


def train(task='reg', config='all', scoring=None, cv=5, n_jobs=None, verbose=1, random_state=None):
    return ModelTrainer(task=task,
                        config=config,
                        scoring=scoring,
                        cv=cv,
                        n_jobs=n_jobs,
                        verbose=verbose,
                        random_state=random_state)


def stack_model(layers, task='reg', stack_method='auto', passthrough=False, cv=None, n_jobs=None, verbose=0):
    assert task in {'reg', 'classif'}

    if task == 'reg':
        return ModelStackRegressor(layers=layers, stack_method=stack_method, passthrough=passthrough,
                                   cv=cv, n_jobs=n_jobs, verbose=verbose)
    else:
        return ModelStackClassifier(layers=layers, stack_method=stack_method, passthrough=passthrough,
                                    cv=cv, n_jobs=n_jobs, verbose=verbose)
