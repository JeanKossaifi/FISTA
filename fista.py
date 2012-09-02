"""
Module implementing the FISTA algorithm
"""
from __future__ import division

__author__ = 'Jean KOSSAIFI'
__licence__ = 'BSD'

import numpy as np
import sys
from scipy.linalg import norm
from math import sqrt
from sklearn.base import BaseEstimator
from sklearn.datasets.base import Bunch
from sklearn.metrics import roc_curve, auc
from hashlib import sha1


def mixed_norm(coefs, p, q=None, n_samples=None, n_kernels=None):
    """ Computes the (p, q) mixed norm of the vector coefs

    Parameters
    ----------
    coefs : ndarray
        a vector indexed by (l, m)
        with l in range(0, n_kernels)
            and m in range(0, n_samples)

    p : int or np.inf

    q : int or np.int

    n_samples : int, optional
        number of elements in each kernel
        default is None

    n_kernels : int, optional
        number of kernels
        default is None

    Returns
    -------
    float
    """
    if q is None or p == q:
        return norm(coefs, p)
    else:
        return norm([norm(i, p) for i in coefs.reshape(
            n_kernels, n_samples)], q)


def dual_mixed_norm(coefs, n_samples, n_kernels, norm_):
    """ Returns a function corresponding to the dual mixt norm

    Parameters
    ----------
    coefs : ndarray
        a vector indexed by (l, m)
        with l in range(0, n_kernels)
            and m in range(0, n_samples)

    n_samples : int, optional
        number of elements in each kernel
        default is None

    n_kernels : int, optional
        number of kernels
        default is None

    norm_ : {'l11', 'l12', 'l21', 'l22'}
        the dual mixed norm we want to compute

    Returns
    -------
    float
    """
    if norm == 'l11':
        res = norm(coefs, np.inf)
    elif norm == 'l12':
        res = mixed_norm(coefs, np.inf, 2, n_samples, n_kernels)
    elif norm == 'l21':
        res = mixed_norm(coefs, 2, np.inf, n_samples, n_kernels)
    else:
        res = norm(coefs, 2)
    return res


def by_kernel_norm(coefs, p, q, n_samples, n_kernels):
    """ Computes the (p, q) norm of coefs for each kernel

    Parameters
    ----------
    coefs : ndarray
        a vector indexed by (l, m)
        with l in range(0, n_kernels)
            and m in range(0, n_samples)

    p : int or np.inf

    q : int or np.inf

    n_samples : int, optional
        number of elements in each kernel
        default is None

    n_kernels : int, optional
        number of kernels
        default is None

    Returns
    -------
    A list of the norms of the sub vectors associated to each kernel
    """
    return [mixed_norm(i, p, q, n_samples, 1)
            for i in coefs.reshape(n_kernels, n_samples)]


def prox_l11(u, lambda_):
    """ Proximity operator for l(1, 1, 2) norm

    Parameters
    ----------
    u : ndarray
        The vector (of the n-dimensional space) on witch we want
        to compute the proximal operator

    lambda_ : float
        regularisation parameter

    Returns
    -------
    ndarray : the vector corresponding to the application of the
             proximity operator to u

    Notes
    -----

    .. math::

       \\hat{\\alpha}_{\\ell,m} = \\sign(u_{\\ell,m})\\left||u_{\\ell,m}| - \\lambda \\right|_+

    """
    return np.sign(u) * np.maximum(np.abs(u) - lambda_, 0.)

def prox_l22(u, lambda_):
    """ proximity operator l(2, 2, 2) norm

    Parameters
    ----------

     u : ndarray
        The vector (of the n-dimensional space) on witch we want to compute the proximal operator

    lambda_ : float
        regularisation parameter

    Returns
    -------

    ndarray : the vector corresponding to the application of the proximity operator to u

    Notes
    -----

    .. math::

       \hat{\alpha}_{\ell,m} = \frac{1}{1 + \lambda} \, u_{\ell,m}

    """
    return 1./(1.+lambda_)*u

def prox_l21_1(u, l, n_samples, n_kernels):
    """ Proximity operator l(2, 1, 1) norm

    Parameters
    ----------
    u : ndarray
        The vector (of the n-dimensional space) on witch we want to compute the proximal operator

    lambda_ : float
        regularisation parameter
    
    n_samples : int, optional
        number of elements in each kernel
        default is None

    n_kernels : int, optional
        number of kernels
        default is None

    Returns
    -------
    ndarray : the vector corresponding to the application of the proximity operator to u


    Notes
    -----
    
    .. math::

       \hat{\alpha}_{\ell,m} = u_{\ell,m} \left| 1 - \frac{\lambda}{\|\bu_{\ell\bullet}\|_{2}} \right|_+\

    where l is in range(0, n_samples) and m is in range(0, n_kernels)
    so :math:`u_{\ell\bullet}` = [u(l, m) for m in n_kernels]

    """
    return (u.reshape(n_kernels, n_samples) *\
        [max(1. - l/norm(u[np.arange(n_kernels)*n_samples+i], 2), 0.)
            for i in range(n_samples)]).reshape(-1)


def prox_l21(u, l, n_samples, n_kernels):
    """ proximity operator l(2, 1, 2) norm

    Parameters
    ----------
    u : ndarray
        The vector (of the n-dimensional space) on witch we want to compute the proximal operator

    lambda_ : float
        regularisation parameter

    n_samples : int, optional
        number of elements in each kernel
        default is None

    n_kernels : int, optional
        number of kernels
        default is None


    Returns
    -------
    ndarray : the vector corresponding to the application of the proximity operator to u

    Notes
    -----

    .. math::

       \hat{\alpha}_{\ell,m} = u_{\ell,m} \left| 1 - \frac{\lambda}{\|\bu_{\ell\bullet}\|_{2}} \right|_+\

    where l is in range(0, n_kernels) and m is in range(0, n_samples)
    so :math:`u_{\ell\bullet}` = [u(l, m) for l in n_samples]

    """
    for i in u.reshape(n_kernels, n_samples):
        n = norm(i, 2)
        if n==0 or n==np.Inf:
            i[:] = 0
        else:
            i[:] *=  max(1. - l/n, 0.)
        # !! If you do just i *= , u isn't modified
        # The slice is needed here so that the array can be modified
    return u


def prox_l12(u, l, n_samples, n_kernels):
    """ proximity operator for l(1, 2, 2) norm

    Parameters
    ----------
    u : ndarray
        The vector (of the n-dimensional space) on witch we want to compute the proximal operator

    lambda_ : float
        regularisation parameter

    n_samples : int, optional
        number of elements in each kernel
        default is None

    n_kernels : int, optional
        number of kernels
        default is None

    Returns
    -------
    ndarray : the vector corresponding to the application of the proximity operator to u


    Notes
    -----

    .. math::

       \hat{\alpha}_{\ell,m} = \sign(u_{\ell,m})\left||u_{\ell,m}| - \frac{\lambda \sum\limits_{m_\ell=1}^{M_\ell} \check u_{\ell,m_\ell}}{(1+\lambda M_\ell) \|\bu_{\ell\bullet}\|_{2}} \right|_+

    where  :math:`\check  u_{\ell,m_\ell}`  denotes the :math:`|u_{\ell,m_\ell}|`
        ordered  by descending  order for fixed  :math:`\ell`,  and the
            quantity :math:`M_\ell` is the number computed in compute_M

    """
    for i in u.reshape(n_kernels, n_samples):
        Ml, sum_Ml = compute_M(i, l, n_samples)
        # i[:] so that u is really modified
        n = norm(i, 2)
        if n == 0 or n == np.Inf:
            i[:] = 0
        else:
            i[:] = np.sign(i)*np.maximum(
                np.abs(i)-(l*sum_Ml)/((1.+l*Ml)*n), 0.)
    return u

def compute_M(u, lambda_, n_samples):
    """
    Parameters
    ----------
    u : ndarray 
        ndarray of size (n_samples * n_samples) representing a subvector of K,
        ie the samples for a single kernel

    lambda_ : int

    n_samples : int
        number of elements in each kernel 
        ie number of elements of u

    Notes
    -----
    
    :math:`M_\ell` is the number such that

    .. math::

       \check u_{\ell,M_\ell+1} \leq  \lambda \sum_{m_\ell=1}^{M_\ell+1} \left(\check u_{\ell,m_\ell} - \check u_{\ell,M_\ell+1}\right) \ ,

    and

    .. math::

       \check     u_{\ell,M_\ell}    >    \lambda\sum_{m_\ell=1}^{M_\ell} \left(\check u_{\ell,m_\ell} - \check u_{k,M_\ell}\right)\ .

    Explication
    -----------
    let u denotes |u(l)|, the vector associated with the kernel l, ordered by descending order
    Ml is the integer such that
        u(Ml) <= l * sum(k=1..Ml + 1) (u(k) - u(Ml + 1))    (S1)
        and
        u(Ml) > l * sum(k=1..Ml) (u(k) - u(Ml)              (S2)
    Note that in that definition, Ml is in [1..Ml]
    In python, while Ml is in [1..(Ml-1)], indices will be in [0..(Ml-1)], so we must take care of indices.
    That's why, we consider Ml is in [0..(Ml-1)] and, at the end, we add 1 to the result

    Example
    -------
    if u(l) = [0 1 2 3] corrsponds to the vector associated with a kernel
        then u = |u(l)| ordered by descending order ie u = [3 2 1 0]

    Then u = [3 2 1 0]
    let l = 1
    Ml is in {0, 1, 2} (not 3 because we also consider Ml+1)
    # Note : in fact Ml is in {1, 2, 3} but it is more convenient
    # to consider it is in {0, 1, 2} as indexing in python starts at 0
    # We juste have to add 1 to the final result

    if Ml = 0 then S1 = 1 and S2 = 0
    if Ml = 1 then S1 = 3 and S2 = 1
    if Ml = 2 then S1 = 6 and S2 = 3

    if Ml = 0 then u(Ml+1)=u(1)=2  > l*... =1  (S1 is not verified)
              and  u(Ml)=u(0)=3    > l*... =0  (S2 is verified)

    if Ml = 1 then u(Ml+1)=u(2)=1 <= l*... =3  (S1 is verified)
              and  u(Ml)=u(1)=2    > l*... =1  (S2 is verified)

    if Ml = 2 then u(Ml+1)=u(3)=0 <= l*... =6  (S1 is verified)
              but  u(Ml)=u(2)=1   <= l*... =3  (S1 is not verified)

    Conclusion : Ml = 1 + 1 !!
    Ml = 2 because in python, indexing starts at 0, so Ml +1

    """
    u = np.sort(np.abs(u))[::-1]
    S1 = u[1:] - lambda_*(np.cumsum(u)[:-1] - (np.arange(n_samples-1)+1)*u[1:])
    S2 = u[:-1] - lambda_*(np.cumsum(u)[:-1] - (np.arange(n_samples-1)+1)*u[:-1])
    Ml = np.argmax((S1<=0.) & (S2>0.)) + 1.

    return Ml, np.sum(u[:Ml]) # u[:Ml] = u[0, 1, ..., Ml-1] !!


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
    return np.dot(K.transpose(), y - np.dot(K,Z))


def _load_Lipschitz_constant(K):
    """ Loads the Lipschitz constant and computes it if not already saved

    Parameters
    ----------
    K : 2D-ndarray
        The matrix of witch we want to compute the Lipschitz constant

    Returns
    -------
    float

    Notes
    -----
    Lipshitz constant is just a number < 2/norm(np.dot(K, K.T), 2)

    The constant is stored in a npy hidden file, in the current directory.
    The filename is the sha1 hash of the ndarray

    """
    try:
        mu = np.load('./.%s.npy' % sha1(K).hexdigest())
    except:
        mu = 1/norm(np.dot(K, K.transpose()), 2)
        np.save('./.%s.npy' % sha1(K).hexdigest(), mu)
    return mu
    

class Fista(BaseEstimator):
    """


    Fast iterative shrinkage/thresholding Algorithm

    Parameters
    ----------

    lambda_ : int, optionnal
        regularisation parameter
        default is 0.5

    loss : {'squared-hinge', 'least-square'}, optionnal
        the loss function to use
        defautl is 'squared-hinge'
        
    penalty : {'l11', 'l22', 'l12', 'l21'}, optionnal
        norm to use as penalty
        default is l11

    n_iter : int, optionnal
        number of iterations
        default is 1000

    recompute_Lipschitz_constant : bool, optionnal
        if True, the Lipschitz constant is recomputed everytime
        if False, it is stored based on it's sha1 hash
        default is False

    """
    
    def __init__(self, lambda_=0.5, loss='squared-hinge', penalty='l11', n_iter=1000, recompute_Lipschitz_constant=False):
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.loss = loss
        self.penalty = penalty
        self.p = int(penalty[1])
        self.q = int(penalty[2])
        self.recompute_Lipschitz_constant = recompute_Lipschitz_constant

    def fit(self, K, y, Lipschitz_constant=None,  verbose=0, **params):
        """ Fits the estimator

        We want to solve a problem of the form y = KB + b
            where K is a (n_samples, n_kernels*n_samples) matrix.

        Parameters
        ---------
        K : ndarray
            numpy array of shape (n, p)
            K is the concatenation of the p/n kernels
                where each kernel is of size (n, n)

        y : ndarray
            an array of the labels to predict for each kernel
            y is of size p
                where K.shape : (n, p)

        Lipschitz_constant : float, optionnal
             allow the user to pre-compute the Lipschitz constant
             (its computation can be very slow, so that parameter is very
             usefull if you were to use several times the algorithm on the same data)

        verbose : {0, 1}, optionnal
            verbosity of the method : 1 will display informations while 0 will display nothing
            default = 0

        Returns
        -------
        self
        """
        next_step = hinge_step
        if self.loss=='squared-hinge':
            K = y[:, np.newaxis] * K
            # Equivalent to K = np.dot(np.diag(y), X) but faster
        elif self.loss=='least-square':
            next_step = least_square_step

        (n_samples, n_features) = K.shape
        n_kernels = n_features/n_samples # We assume each kernel is a square matrix
        self.n_samples, self.n_kernels = n_samples, n_kernels

        if Lipschitz_constant==None:
            Lipschitz_constant = _load_Lipschitz_constant(K)

        tol = 10**(-6)
        coefs_current = np.zeros(n_features, dtype=np.float) # coefficients to compute
        coefs_next = np.zeros(n_features, dtype=np.float)
        Z = np.copy(coefs_next) # a linear combination of the coefficients of the 2 last iterations
        tau_1 = 1

        if self.penalty=='l11':
            prox = lambda(u):prox_l11(u, self.lambda_*Lipschitz_constant)
        elif self.penalty=='l22':
            prox = lambda(u):prox_l22(u, self.lambda_*Lipschitz_constant)
        elif self.penalty=='l21':
            prox = lambda(u):prox_l21(u, self.lambda_*Lipschitz_constant, n_samples, n_kernels)
        elif self.penalty=='l12':
            prox = lambda(u):prox_l12(u, self.lambda_*Lipschitz_constant, n_samples, n_kernels)

        if verbose==1:
            self.iteration_dual_gap = list()

        for i in range(self.n_iter):
            coefs_current = coefs_next # B_(k-1) = B_(k)
            coefs_next = prox(Z + Lipschitz_constant*next_step(y, K, Z))
            
            tau_0 = tau_1 #tau_(k+1) = tau_k
            tau_1 = (1 + sqrt(1 + 4*tau_0**2))/2

            Z = coefs_next + (tau_0 - 1)/tau_1*(coefs_next - coefs_current)
            
            # Dual problem
            objective_var = 1 - np.dot(K, coefs_next)
            objective_var = np.maximum(objective_var, 0) # Shrink
            # Primal objective function
            penalisation = 0.5*self.lambda_/self.q*(mixed_norm(coefs_next,
                    self.p, self.q, n_samples, n_kernels)**self.q)
            loss = np.sum(objective_var**2)
            objective_function = penalisation + loss
            # Dual objective function
            dual_var = objective_var
            if self.lambda_ != 0:
                #dual_penalisation = dual_mixed_norm(np.dot(K.T,dual_var)/self.lambda_,
                dual_penalisation = dual_mixed_norm(self.lambda_/self.q*np.dot(K.T,dual_var),
                        n_samples, n_kernels, self.penalty)
                if self.q==1:
                    # Fenchel conjugate of a mixed norm
                    if dual_penalisation > 1:
                        dual_var = dual_var / dual_penalisation
                    dual_penalisation = 0
                else:
                    # Fenchel conjugate of a squared mixed norm
                    #dual_penalisation = 0.5*self.lambda_*(dual_penalisation**2)
                    dual_penalisation = 0.5*(dual_penalisation**2)
            else:
                dual_penalisation = 0
            dual_loss = -0.5*np.sum(dual_var**2) + np.dot(dual_var, y)#np.sum(dual_var)
            # trace(np.dot(duat_var[:, np.newaxis], y)) au lieu du sum(dual_var) ?
            dual_objective_function = dual_loss - dual_penalisation
            gap = abs(objective_function - dual_objective_function)

            if verbose:
                sys.stderr.write("Iteration : %d\r" % i )
                # print "iteration %d" % i
                self.iteration_dual_gap.append(gap)
                if i%1000 == 0:
                    print "primal objective : %f, dual objective : %f, dual_gap : %f" % (objective_function, dual_objective_function, gap)

            if gap<=tol and i>10:
                print "convergence at iteration : %d" %i
                break

        if verbose:
            print "dual gap : %f" % gap
            print "objective_function : %f" % objective_function
            print "dual_objective_function : %f" % dual_objective_function
            print "dual_penalisation : %f" % dual_penalisation
            print "dual_loss : %f" % dual_loss
        self.coefs_ = coefs_next
        self.gap = gap
        self.objective_function = objective_function
        self.dual_objective_function = dual_objective_function

        return self

    def predict(self, K):
        """ Returns the prediction associated to the Kernels represented by K

        Parameters
        ----------
        K : ndarray 
            ndarray of size (n_samples, n_kernels*n_samples) representing the kernels

        Returns
        -------
        ndarray : the prediction associated to K
        """
        if self.loss=='squared-hinge':
            return np.sign(np.dot(K, self.coefs_))
        else:
            return np.dot(K, self.coefs_)

    def score(self, K, y):
        """ Returns the score prediction for the given data

        Parameters
        ----------
        K : ndarray
            matrix of observations

        y : ndarray
            the labels correspondings to K

        Returns
        -------
        The percentage of good classification for K
        """
        if self.loss=='squared-hinge':
            return np.sum(np.equal(self.predict(K), y))*100./len(y)
        else:
            print "Score not yet implemented for regression\n"
            return None

    def info(self, K, y):
        """ For test purpose

        Parameters
        ----------
        K : 2D-array
            kernels

        y : ndarray
            labels
        Returns
        -------
        A dict of informations
        """
        result = Bunch()
        n_samples, n_kernels = self.n_samples, self.n_kernels
        nulled_kernels = 0
        nulled_coefs_per_kernel = list()

        for i in self.coefs_.reshape(n_kernels, n_samples):
            if len(i[i!=0]) == 0:
                nulled_kernels = nulled_kernels + 1
            nulled_coefs_per_kernel.append(len(i[i==0]))

        result['score'] = self.score(K, y)
        result['norms'] = by_kernel_norm(self.coefs_, self.p, self.q,
                n_samples, n_kernels)
        result['nulled_coefs'] = len(self.coefs_[self.coefs_==0])
        result['nulled_kernels'] = nulled_kernels
        result['nulled_coefs_per_kernel'] = nulled_coefs_per_kernel
        result['objective_function'] = self.objective_function
        result['dual_objective_function'] = self.dual_objective_function
        result['gap'] = self.gap
        fpr, tpr, thresholds = roc_curve(y, self.predict(K))
        result['ROC.fpr'] = fpr
        result['ROC.tpr'] = tpr
        result['ROC.thresholds'] = thresholds
        result['auc'] = auc(fpr, tpr)
        result['lambda_'] = self.lambda_
        
        return result
