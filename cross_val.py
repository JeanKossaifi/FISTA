from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.datasets.base import Bunch
from sklearn import clone

import numpy as np

import fista

def _KFold(y, n_folds, n_samples, n_kernels):
    return [(train, np.tile(train, n_kernels), test) for train, test in StratifiedKFold(y, n_folds, indices=False)]


def _sub_info(estimator, K_train, y_train, K_test, y_test, lambda_=None):
    if lambda_ is not None:
        estimator.lambda_ = lambda_
    return estimator.fit(K_train, y_train).info( K_test, y_test)


def _compute_info(estimator, K, y, lambda_, folds):
    estimator.lambda_ = lambda_
    # Cross_validation loop
    infos = Parallel(n_jobs=1, verbose=3)(
            delayed(_sub_info)(clone(estimator),
                 K[train_lines, :][:, train_columns], y[train_lines],
                 K[test, :][:, train_columns], y[test])
           for (train_lines, train_columns, test) in folds)

    scores = [i['score'] for i in infos]
    return infos, np.mean(scores), np.std(scores)


def cross_val(penalty, lambda_, n_folds, K, y, n_iter=1000):
    estimator = fista.Fista(penalty=penalty, n_iter=n_iter)
    n_samples, n_features = K.shape
    n_kernels = n_features / n_samples
    folds = _KFold(y, n_folds, n_samples, n_kernels)
    # Computing score mean score for each lambda
    print "** Computing scores ..."
    infos, mean, std = _compute_info(estimator, K, y, lambda_, folds)
    result = Bunch()
    result['infos'] = infos
    result['penalty'] = penalty
    result['lambdas'] = lambda_
    result['std'] = std
    result['mean_score'] = mean
    result['n_folds'] = n_folds

    return result


def intern_cross_val(estimator, lambdas, n_folds_int, n_samples, n_kernels, K, y, train_lines, train_columns, test):
    folds_int = _KFold(y, n_folds_int, n_samples, n_kernels)
    int_scores = np.zeros((len(folds_int), len(lambdas)))
    for j, (train_lines, train_columns, test) in enumerate(folds_int):
        for k, lambda_ in enumerate(lambdas):
            estimator.lambda_ = lambda_
            int_scores[j, k] = (estimator.fit(
                K[train_lines, :][:, train_columns], y[train_lines]).score(
                    K[test, :][:, train_columns], y[test]))
    int_scores = np.mean(int_scores, axis=0)
    print "score moyen de chaque lambda :", int_scores
    best_lambda = np.argmax(int_scores)
    int_score = int_scores[best_lambda]
    best_lambda = lambdas[best_lambda]
    estimator.lambda_ = best_lambda
    ext_score = estimator.fit(
            K[train_lines, :][:, train_columns], y[train_lines]).score(
                K[test, :][:, train_columns], y[test])
    return {'best lambda' : best_lambda, 'int score' : int_score, 'ext score' : ext_score}


def double_cross_val(penalty, lambdas, n_folds_ext, n_folds_int, K, y, n_iter=1000, n_jobs=-1):
    estimator = fista.Fista(penalty=penalty, n_iter=n_iter)
    n_samples, n_features = K.shape
    n_kernels = n_features / n_samples
    folds_ext = _KFold(y, n_folds_ext, n_samples, n_kernels)
    # Computing score mean score for each lambda
    print "** Computing scores ..."
    best_results = Parallel(n_jobs=n_jobs, verbose=51)\
            (delayed(intern_cross_val)\
            (clone(estimator), lambdas, n_folds_int, n_samples, n_kernels, K, y, train_lines, train_columns, test)
               for (train_lines, train_columns, test) in folds_ext)

    return best_results


def lambda_choice(penalty, lambdas, n_folds, K, y, n_iter=1000, verbose=0, n_jobs=-1):
    estimator = fista.Fista(penalty=penalty, n_iter=n_iter)
    infos = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_sub_info)(clone(estimator), K, y, K, y, lambda_)
           for lambda_ in lambdas)

    return infos
