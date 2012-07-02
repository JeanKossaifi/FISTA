"""
Module implementing the FISTA algorithm
"""

__author__ = 'Jean KOSSAIFI'
__licence__ = 'BSD'

import numpy as np
from scipy.linalg import norm
from math import sqrt

def prox_l11(u, l):
    return np.sign(u)*np.maximum(np.abs(u) - l, 0.)

def prox_l22(u, l):
    return 1./(1.+l)*u

def prox_l21(u, l, n_samples, n_kernels):
    res = np.array([max(1. - l/norm(u[np.arange(n_kernels)*n_samples+i], 2), 0.) for i in range(n_samples)])
    for i in range(n_kernels-1):
        res = np.concatenate((res, res))
    return u*res

def fista(K, y, l, penalty='l11', n_iter=500):
    """
    We want to solve a problem of the form y = XB + b
        where X is a (n, p) matrix.

    arguments
    ---------
    K : 2-D numpy array of shape (n, p)
        K is the concatenation of the p/n kernels
            where each kernel is of size (n, n)

    y : numpy arrays
        an array of the labels to predict for each kernel
        y is of size p
            where K.shape : (n, p)

    penalty : string, optionnal
        default : 'l11'
        possible values : ('l11', 'l22' or 'l21')

    n_iter : int, optionnal
        number of iterations

    return
    ------
    B : 
    coefficient computed
    """
    (n_samples, n_features) = K.shape
    B_0 = B_1 = np.zeros(n_features) # coefficients to compute
    tol = 10**(-5)
    Z = B_1
    tau_1 = 1
    mu = 1/norm(np.dot(K, K.transpose()))
    n_kernels = n_features/n_samples

    if penalty=='l11':
        prox = lambda(u):prox_l11(u, l)
    elif penalty=='l22':
        prox = lambda(u):prox_l22(u, l)
    elif penalty=='l21':
        prox = lambda(u):prox_l21(u, l, n_samples, n_kernels)

    for i in range(n_iter):
        B_0 = B_1 # B_(k-1) = B_(k)
        tau_0 = tau_1 #tau_(k+1) = tau_k
        B_1 = prox(Z + mu*np.dot(K.transpose(), y - np.dot(K,Z)))
        tau_1 = (1 + sqrt(1 + 4*tau_0**2))/2
        Z = B_1 + (tau_0 - 1)/tau_1*(B_1 - B_0)

#        if norm(B_1 - B_0, 2)/norm(B_1,2) <= tol:
#            return B_1

    return B_1


X = np.array([[1, 2, 1, 2, 4, 2],[1, 0, 0, 2, 0, 0], [0, 0, 1, 0, 0, 2]])
B_real = np.array([1, 0, -1])
y = np.array([1, 1, -1])
B = fista(X, y, 0.5, 'l11', n_iter=20)


X2 = np.random.normal(size=(10, 10))
y2 = np.random.normal(size=10)
B2 = fista(X2, y2, 0.5, 'l11', n_iter=100)
print "taux de bonne prediction with l11: %f " % (np.sum(np.equal(np.dot(X2, B2), y2))/10.)
B2 = fista(X2, y2, 0.5, 'l22', n_iter=100)
print "taux de bonne prediction with l22: %f " % (np.sum(np.equal(np.dot(X2, B2), y2))/10.)
B2 = fista(X2, y2, 0.5, 'l21', n_iter=100)
print "taux de bonne prediction with l21: %f " % (np.sum(np.equal(np.dot(X2, B2), y2))/10.)
