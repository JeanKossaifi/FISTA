"""
Module implementing the FISTA algorithm
"""

__author__ = 'Jean KOSSAIFI'
__licence__ = 'BSD'

import numpy as np
from scipy.linalg import norm
from math import sqrt

def prox_l11(u, l):
    """
    l(1, 1, 2) norm

    parameters
    ----------
    u : np-array
        a point of the n-dimensional space

    l : integer
        regularisation parameter

    returns
    -------
    np-array
    the point corresponding to the application of 
    the proximity operator to u
    """
    return np.sign(u)*np.maximum(np.abs(u) - l, 0.)

def prox_l22(u, l):
    """
    l(2, 2, 2) norm, see prox_l11
    """
    return 1./(1.+l)*u

def prox_l21_1(u, l, n_samples, n_kernels):
    """
    l(2, 1, 1) norm, see prox_l11
    """
    res = np.array([max(1. - l/norm(u[np.arange(n_kernels)*n_samples+i], 2), 0.) for i in range(n_samples)])
    for i in range(n_kernels-1):
        res = np.concatenate((res, res))
    return u*res

def prox_l21(u, l, n_samples, n_kernels):
    """
    l(2, 1, 2) norm, see prox_l11
    """
    res = np.zeros(n_samples*n_kernels)
    for i in range(n_kernels):
        res[i*n_samples:(i+1)*n_samples] =\
                max(1. - l/norm(u[i*n_samples:(i+1)*n_samples], 2), 0.)
    return u*res

def hinge_step(y, K, Z):
    """
    Returns the point in witch we apply gradient descent

    parameters
    ----------
    y : np-array
        the labels vector

    K : 2D np-array
        the concatenation of all the kernels, of shape
        n_samples, n_kernels*n_samples

    Z : a linear combination of the last two coefficient vectors

    returns
    -------
    res : np-array of shape n_samples*,_kernels
          a point of the space where we will apply gradient descent
    """
    return np.dot(K.transpose(), np.maximum(1 - np.dot(K, Z), 0))

def least_square_step(y, K, Z):
    return np.dot(K.transpose(), y - np.dot(K,Z))
    
def fista(K, y, l, loss='hinge', penalty='l11', n_iter=500):
    """
    We want to solve a problem of the form y = KB + b
        where K is a (n_samples, n_kernels*n_samples) matrix.

    arguments
    ---------
    K : 2-D numpy array of shape (n, p)
        K is the concatenation of the p/n kernels
            where each kernel is of size (n, n)

    y : numpy arrays
        an array of the labels to predict for each kernel
        y is of size p
            where K.shape : (n, p)

    loss : string, optionnal
        defautl : 'hinge'
        possible values : 'hinge' or 'least-square'
        
    penalty : string, optionnal
        default : 'l11'
        possible values : 'l11', 'l22' or 'l21'

    n_iter : int, optionnal
        number of iterations

    return
    ------
    B : ndarray
    coefficient computed
    """
    
    step = hinge_step
    if loss=='hinge':
        K = np.dot(np.diag(y), K)
    elif loss=='least-square':
        step = least_square_step

    (n_samples, n_features) = K.shape
    B_0 = B_1 = np.zeros(n_features) # coefficients to compute
    tol = 10**(-5)
    Z = B_1
    tau_1 = 1
    mu = 1/norm(np.dot(K, K.transpose()))
    n_kernels = n_features/n_samples

    if penalty=='l11':
        prox = lambda(u):prox_l11(u, l*mu)
    elif penalty=='l22':
        prox = lambda(u):prox_l22(u, l*mu)
    elif penalty=='l21':
        prox = lambda(u):prox_l21(u, l*mu, n_samples, n_kernels)

    for i in range(n_iter):
        B_0 = B_1 # B_(k-1) = B_(k)
        tau_0 = tau_1 #tau_(k+1) = tau_k
        B_1 = prox(Z + mu*step(y, K, Z))
        tau_1 = (1 + sqrt(1 + 4*tau_0**2))/2
        Z = B_1 + (tau_0 - 1)/tau_1*(B_1 - B_0)

        if norm(B_1 - B_0, 2)/norm(B_1,2) <= tol:
            print "convergence at iteration : %d" % i
            return B_1

    return B_1


X = np.array([[1, 2, 1, 2, 4, 2],[1, 0, 0, 2, 0, 0], [0, 0, 1, 0, 0, 2]])
B_real = np.array([1, 0, -1])
y = np.array([1, 1, -1])
B = fista(X, y, 0.5, penalty='l11', n_iter=20)


X2 = np.random.normal(size=(10, 40))
y2 = np.sign(np.random.normal(size=10))
B2 = fista(X2, y2, 0.5, penalty='l11', n_iter=1000)
print "taux de bonne prediction with l11: %f " % (np.sum(np.equal(np.sign(np.dot(X2, B2)), y2))/10.)
B2 = fista(X2, y2, 0.5, penalty='l22', n_iter=1000)
print "taux de bonne prediction with l22: %f " % (np.sum(np.equal(np.sign(np.dot(X2, B2)), y2))/10.)
B2 = fista(X2, y2, 0.5, penalty='l21', n_iter=1000)
print "taux de bonne prediction with l21: %f " % (np.sum(np.equal(np.sign(np.dot(X2, B2)), y2))/10.)
