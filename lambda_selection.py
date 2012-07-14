from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.datasets.base import Bunch
from sklearn import clone

import numpy as np

from scipy.linalg import norm

import fista

def _compute_mu(K, n_folds):
    return 1/norm(np.dot(K, K.transpose()), 2)

def _KFold(y, n_folds, n_samples, n_kernels):
    return [(train, np.tile(train, n_kernels), test) for train, test in StratifiedKFold(y, n_folds)]

def _compute_score(estimator, K, y, lambda_, mus, n_folds, kf, n_samples, n_kernels):
    scores = np.zeros(n_folds)
    estimator.lambda_ = lambda_
    # Cross_validation loop
    for i, (train_lines, train_columns, test) in enumerate(kf):
        # computing the score
        scores[i] = estimator.fit(K[train_lines, :][:, train_columns], y[train_lines, :], verbose=1).score(
                K[test, :][:, train_columns], y[test, :])

    return np.mean(scores)

def select_lambda(penalty, lambdas, n_folds, K, y, K_name='default'):
    estimator = fista.Fista(penalty=penalty, n_iter=10)
    n_samples, n_features = K.shape
    n_kernels = n_features / n_samples
    folds = _KFold(y, n_folds, n_samples, n_kernels)
    # Computing the coefficients mus for each fold
    print "** Computing the coefficients mus..."
    try:
        mus = np.load('./mus_%s_kernel__%d_folds.npy' % (K_name, n_folds))
    except IOError:
        print "computing the mus for scratch ..."
        mus = Parallel(n_jobs=-1, verbose=2)\
                (delayed(_compute_mu)(K[train_lines, :][:, train_columns], n_folds)\
            for train_lines, train_columns, test in folds)
        mus = np.array(mus)
        np.save('mus_%s_kernel__%d_folds.npy' % (K_name, n_folds), mus)
    print "\n\n** ... MUS : DONE\n\n"
    # Computing score mean score for each lambda
    print "** Computing scores ..."
    scores = Parallel(n_jobs=1, verbose=2)\
            (delayed(_compute_score)\
            (clone(estimator), K, y, i, mus, n_folds, folds, n_samples, n_kernels)\
             for i in lambdas) 
    result = Bunch()
    result['scores'] = dict(zip(lambdas, scores))
    result['penalty'] = penalty
    result['lambdas'] = lambdas
    result['lambda_scores'] = scores
    result['n_folds'] = n_folds

    return result
