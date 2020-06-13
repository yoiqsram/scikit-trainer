from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier
from sktrainer import train, stack

X, y = load_digits(return_X_y=True)

if __name__ == '__main__':
    model = train([('dt', {}), ('tb', {})], n_jobs=-1).fit(X, y)

    model_ = stack([model.estimators_, ExtraTreesClassifier()]).fit(X, y)

    print('Stacked score:', model_.score(X, y))
