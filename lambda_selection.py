from sklearn.cross_validation import KFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.datasets.base import Bunch
from sklearn import clone

import numpy as np

from scipy.linalg import norm

import fista

def _compute_mu(K, n_folds):
    return 1/norm(np.dot(K, K.transpose()), 2)

def compute_score(estimator, K, y, lambda_, mus, n_folds, kf, n_samples, n_kernels):
    scores = np.zeros(n_folds)
    estimator.lambda_ = lambda_
    # Cross_validation loop
    for i, (train, test) in enumerate(kf):
        train_col = list()
        # little hack to have squared kernels
        for i in range(n_kernels):
            train_col.extend(train+n_samples*i)
#        test_col = list()
#        for i in range(n_kernels):
#            test_col.extend(test+n_samples*i)
        # computing the score
        scores[i] = estimator.fit(K[train, :][:, train_col], y[train, :], mus[i], verbose=1).score(
                K[test, :][:, train_col], y[test, :])

    return np.mean(scores)

def select_lambda(penalty, lambdas, n_folds, K, y, K_name='default'):
    estimator = fista.Fista(penalty=penalty, n_iter=100)
    folds = KFold(K.shape[0], n_folds)
    n_samples, n_features = K.shape
    n_kernels = n_features / n_samples
    # Computing the coefficients mus for each fold
    print "** Computing the coefficients mus..."
    try:
        mus = np.load('./mus_%s_kernel__%d_folds.npy' % (K_name, n_folds))
    except IOError:
        print "computing the mus for scratch ..."
        mus = Parallel(n_jobs=-1, verbose=2)\
            (delayed(_compute_mu)(K[train, :], n_folds)\
            for train, test in folds)
        mus = np.array(mus)
        np.save('mus_%s_kernel__%d_folds.npy' % (K_name, n_folds), mus)
    print "\n\n** ... MUS : DONE\n\n"
    # Computing score mean score for each lambda
    print "** Computing scores ..."
    scores = Parallel(n_jobs=1, verbose=2)\
            (delayed(compute_score)\
            (clone(estimator), K, y, i, mus, n_folds, folds, n_samples, n_kernels)\
             for i in lambdas) 
    result = Bunch()
    result['scores'] = dict(zip(lambdas, scores))
    result['penalty'] = penalty
    result['lambdas'] = lambdas
    result['lambda_scores'] = scores
    result['n_folds'] = n_folds

    return result
