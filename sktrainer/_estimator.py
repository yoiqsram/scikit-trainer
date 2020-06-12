from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor


classifier = {
    'dt': DecisionTreeClassifier,
    'tb': BaggingClassifier,
    'rf': RandomForestClassifier,
    'et': ExtraTreesClassifier,
    'ab': AdaBoostClassifier,
    'gb': GradientBoostingClassifier,
    'logreg': LogisticRegression,
    'lda': LinearDiscriminantAnalysis,
    'qda': QuadraticDiscriminantAnalysis,
    'svm': SVC,
    'mlp': MLPClassifier
}

regressor = {
    'dt': DecisionTreeRegressor,
    'tb': BaggingRegressor,
    'rf': RandomForestRegressor,
    'et': ExtraTreesRegressor,
    'ab': AdaBoostRegressor,
    'gb': GradientBoostingRegressor,
    'linreg': LinearRegression,
    'ridge': Ridge,
    'svm': SVR,
    'mlp': MLPRegressor
}
