from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.datasets.base import Bunch
from sklearn import clone

import numpy as np

from scipy.linalg import norm

import fista

def _compute_mu(K, n_folds):
    return 1/norm(np.dot(K, K.transpose()), 2)

def _load_mu(K, n_folds, folds, K_name):
    # Computing the coefficients mus for each fold
    print "** Computing the coefficients mus..."
    try:
        mus = np.load('./mus_%s_kernel__%d_folds.npy' % (K_name, n_folds))
    except IOError:
        print "computing the mus for scratch ..."
        mus = Parallel(n_jobs=-1, verbose=2)\
               (delayed(_compute_mu)(K[train_lines, :][:, train_columns],
                   n_folds) for train_lines, train_columns, test in folds)
        mus = np.array(mus)
        np.save('mus_%s_kernel__%d_folds.npy' % (K_name, n_folds), mus)
    print "\n\n** ... MUS : DONE\n\n"
    return mus

def _KFold(y, n_folds, n_samples, n_kernels):
    return [(train, np.tile(train, n_kernels), test) for train, test in StratifiedKFold(y, n_folds)]

def _sub_score(estimator, K_train, y_train, mu, K_test, y_test, file_name):
    return estimator.fit(K_train, y_train, mu, verbose=1).score(
                            K_test, y_test, file_name)

def _compute_save_score(estimator, K, y, lambda_, mus, n_folds, kf, K_name):
    estimator.lambda_ = lambda_
    file_name = "save_estimator_cross_validation__lambda_%f__n_folds_%d___norm_%s__kernel_%s" % (lambda_, n_folds, estimator.penalty, K_name)
    # Cross_validation loop
    scores = Parallel(n_jobs=2, verbose=3)(
            delayed(_sub_score)(clone(estimator),
                 K[train_lines, :][:, train_columns], y[train_lines],
                 mus[i], K[test, :][:, train_columns], y[test],
                 file_name+"%d" % i)
           for i, (train_lines, train_columns, test) in enumerate(kf))

    return np.mean(scores), np.std(scores), scores

def _compute_score(estimator, K, y, lambda_, mus, n_folds, kf):
    scores = np.zeros(n_folds)
    estimator.lambda_ = lambda_
    # Cross_validation loop
    for i, (train_lines, train_columns, test) in enumerate(kf):
        # computing the score
        scores[i] = estimator.fit(K[train_lines, :][:, train_columns], y[train_lines], verbose=1).score(
                K[test, :][:, train_columns], y[test])

    return np.mean(scores)

def select_lambda(penalty, lambdas, n_folds, K, y, K_name='default'):
    estimator = fista.Fista(penalty=penalty, n_iter=2000)
    n_samples, n_features = K.shape
    n_kernels = n_features / n_samples
    folds = _KFold(y, n_folds, n_samples, n_kernels)
    # Loading mus
    mus = _load_mu(K, n_folds, folds, K_name)
    # Computing score mean score for each lambda
    print "** Computing scores ..."
    scores = Parallel(n_jobs=1, verbose=2)\
            (delayed(_compute_score)\
            (clone(estimator), K, y, i, mus, n_folds, folds)
             for i in lambdas) 
    result = Bunch()
    result['scores'] = dict(zip(lambdas, scores))
    result['penalty'] = penalty
    result['lambdas'] = lambdas
    result['lambda_scores'] = scores
    result['n_folds'] = n_folds

    return result

def cross_val(penalty, lambda_, n_folds, K, y, K_name='default'):
    estimator = fista.Fista(penalty=penalty, n_iter=3000)
    n_samples, n_features = K.shape
    n_kernels = n_features / n_samples
    folds = _KFold(y, n_folds, n_samples, n_kernels)
    # Loading the coefficients mus for each fold
    mus = _load_mu(K, n_folds, folds, K_name)
    # Computing score mean score for each lambda
    print "** Computing scores ..."
    mean, std, scores = _compute_save_score(estimator, K, y, lambda_, mus, n_folds, folds, K_name)
    result = Bunch()
    result['score'] = (lambda_, mean)
    result['detailed_scores'] = scores
    result['penalty'] = penalty
    result['lambdas'] = lambda_
    result['std'] = std
    result['mean_score'] = mean
    result['n_folds'] = n_folds

    return result

