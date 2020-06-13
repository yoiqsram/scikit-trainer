from sktrainer._train import *
from sktrainer._stack import *


def train(config, type='classifier', scoring=None, cv=None, n_jobs=None,
          verbose=1, pre_dispatch='2*n_jobs', random_state=None):
    assert type in {'classifier', 'regressor'}

    if type == 'classifier':
        return ModelClassifier(config=config, scoring=scoring, cv=cv, n_jobs=n_jobs,
                               verbose=verbose, pre_dispatch=pre_dispatch, random_state=random_state)
    else:
        return ModelRegressor(config=config, scoring=scoring, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, pre_dispatch=pre_dispatch, random_state=random_state)


def stack(layers, type='classifier', stack_method='auto', passthrough=False, cv=None, n_jobs=None, verbose=0):
    assert type in {'classifier', 'regressor'}

    if type == 'classifier':
        return ModelStackClassifier(layers=layers, stack_method=stack_method, passthrough=passthrough,
                                    cv=cv, n_jobs=n_jobs, verbose=verbose)
    else:
        return ModelStackRegressor(layers=layers, stack_method=stack_method, passthrough=passthrough,
                                   cv=cv, n_jobs=n_jobs, verbose=verbose)
