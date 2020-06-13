from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier
from sktrainer import train, stack_model

X, y = load_digits(return_X_y=True)

if __name__ == '__main__':
    model = train('classif', ['dt', 'tb'], n_jobs=-1).fit(X, y)

    model_ = stack_model([model.estimators_, ExtraTreesClassifier()], 'classif').fit(X, y)

    print('Stacked score:', model_.score(X, y))
