from sklearn.cross_validation import KFold
from sklearn import clone
import numpy as np
from scipy.linalg import norm
from sklearn.externals.joblib import Parallel, delayed
#from sklearn.datasets.base import Bunch

class select_lambda():
    """
    Chooses the best lambda by cross validation
    """
    def __init__(self, estimator, lambdas, n_folds):
        self.estimator = estimator
        self.lambdas = lambdas
        self.n_folds = n_folds

    def fit(self, K, y):
        self.compute_mus(K, self.n_folds)
        scores = Parallel(n_jobs=-1, verbose=1)\
            ( delayed(compute_score)\
            (clone(self.estimator), K, y, i, self.mus, self.n_folds, self.kf)\
             for i in self.lambdas) 
        return dict(zip(self.lambdas, scores))
        
    def compute_mus(self, K, n_folds):
        self.kf = KFold(K.shape[0], n_folds)
        self.mus = np.zeros(n_folds)        
        for i, (train, test) in enumerate(self.kf):
            self.mus[i] = 1/norm(np.dot(K[train, :], K[train, :].transpose()), 2)


def compute_score(estimator, K, y, lambda_, mus, n_folds, kf):
    scores = np.zeros(n_folds)
    estimator.lambda_ = lambda_
    for i, (train, test) in enumerate(kf):
        scores[i] = estimator.fit(K[train, :], y[train, :], mus[i]).score(K[test, :], y[test, :])
    return np.mean(scores)
