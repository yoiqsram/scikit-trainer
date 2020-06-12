from sklearn.datasets import load_iris
from sklearn.ensemble import ExtraTreesClassifier
from sktrainer import train

if __name__ == '__main__':
    X, y = load_iris(True)
    config = {
        'rf': {
            'n_estimators': [10, 25]
        },
        ExtraTreesClassifier: {
            'n_estimators': [10, 25, 50, 100],
            'max_features': ['sqrt', 'log2', None],
            'search': 'random',
            'n_iter': 5
        }
    }
    model = train('classif', config, scoring='accuracy', n_jobs=-1).fit(X, y)
    print()
