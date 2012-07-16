"""
Module implementing the FISTA algorithm
"""
__author__ = 'Jean KOSSAIFI'
__licence__ = 'BSD'

import numpy as np
import sys
from scipy.linalg import norm
from math import sqrt
from sklearn.base import BaseEstimator
from sklearn.datasets.base import Bunch


def norm_l12(u, n_samples, n_kernels):
    """
    Returns the l12 norm of the vector u
    """
    return np.sum(np.sum(np.reshape(np.abs(u), (n_kernels, n_samples)), axis=1)**2)**0.5

def norm_l21(u, n_samples, n_kernels):
    """
    Returns the l21 norm of the vector u
    """
    return np.sum(np.sum(np.reshape(np.abs(u)**2, (n_kernels, n_samples)), axis=1)**0.5)

def prox_l11(u, l):
    """
    proximity operator l(1, 1, 2) norm

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
    proximity operator l(2, 2, 2) norm, see prox_l11
    """
    return 1./(1.+l)*u

def prox_l21_1(u, l, n_samples, n_kernels):
    """
    proximity operator l(2, 1, 1) norm, see prox_l11
    """
    return u*np.tile(np.array([
        max(1. - l/norm(u[np.arange(n_kernels)*n_samples+i], 2), 0.)
            for i in range(n_samples)]), n_kernels)

def prox_l21(u, l, n_samples, n_kernels):
    """
    proximity operator l(2, 1, 2) norm, see prox_l11
    """
    for i in u.reshape(n_kernels, n_samples):
        i *=  max(1. - l/max(norm(i, 2), 0.00000000001), 0.)
    return u


def prox_l12(u, l, n_samples, n_kernels):
    """
    proximity operator for l(1, 2, 2) norm, see prox_l11
    """
    for i in u.reshape(n_kernels, n_samples):
        Ml, sum_Ml = compute_M(i, l, n_samples)
        i = np.sign(i)*np.maximum(
                np.abs(i)-(l*sum_Ml)/((1+l*Ml)*norm(i, 2)), 0)
    return u

def compute_M(u, l, n_samples):
    """
    parameters
    ----------
    u : ndarray of size (n_samples * n_samples)
        subvector for a single kernel

    l : integer

    n_samples : integer
        number of elements in each kernel 
        ie number of elements of u

    explication
    -----------
    let u denotes |u(l)|, the vector associated with the kernel l, ordered by descending order
    Ml is the integer such that
        u(Ml) <= l * sum(k=1..Ml + 1) (u(k) - u(Ml + 1))    (S1)
        and
        u(Ml) > l * sum(k=1..Ml) (u(k) - u(Ml)              (S2)

    example
    -------
    if u(l) = [0 1 2 3] corrsponds to the vector associated with a kernel
        then u = |u(l)| ordered by descending order ie u = [3 2 1 0]

    u = [3 2 1 0]
    let l = 1
    Ml is in [2, 1, 0] (not 3 because we also consider Ml+1)

    if Ml = 0 then S1 = 1 and S2 = 0
    if Ml = 1 then S1 = 3 and S2 = 1
    if Ml = 2 then S1 = 6 and S2 = 3

    if Ml = 0 then u(Ml+1)=u(1)=2  > l*... =1  (S1 is not verified)
              and  u(Ml)=u(0)=3    > l*... =0  (S2 is verified)

    if Ml = 1 then u(Ml+1)=u(2)=1 <= l*... =3  (S1 is verified)
              and  u(Ml)=u(1)=2    > l*... =1  (S2 is verified)

    if Ml = 2 then u(Ml+1)=u(3)=0 <= l*... =6  (S1 is verified)
              but  u(Ml)=u(2)=1   <= l*... =3  (S1 is not verified)

    Conclusion : Ml = 1

    Note 
    ----
    In fact, in the previous example, Ml = 2 because in python, indexing
    starts at 0, so Ml=(Ml + 1)
    """
    u = np.sort(np.abs(u))[::-1]
    S1 = u[1:] - l*(np.cumsum(u)[:-1] - (np.arange(n_samples-1)+1)*u[1:])
    S2 = u[:-1] - l*(np.cumsum(u)[:-1] - (np.arange(n_samples-1)+1)*u[:-1])
    Ml = np.argmax((S1 <= 0) & (S2 > 0)) + 1
    return Ml, np.sum(u[:Ml])


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

def _load_mus(K):
    try:
        mu = np.load('./.%s.npy' % sha1(K))
    except:
        mu = 1/norm(np.dot(K, K.transpose()), 2)
    return mu
    
class Fista(BaseEstimator):
    
    def __init__(self, lambda_=0.5, loss='hinge', penalty='l11', n_iter=500):
        """
        parameters
        ----------

        loss : string, optionnal
            defautl : 'hinge'
            possible values : 'hinge' or 'least-square'
            
        penalty : string, optionnal
            default : 'l11'
            possible values : 'l11', 'l22' or 'l21'

        n_iter : int, optionnal
            number of iterations

        """
        self.n_iter = n_iter
        self.lambda_ = lambda_
        self.loss = loss
        self.penalty = penalty

    def fit(self, K, y, mu=None, verbose=0):
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

        mu : float, optionnal
             allow the user to pre-compute mu
             (the computation of mu can be very slow, so that parameter is very
             usefull if you were to use several times the algorithm on the same data)

        verbose : int, optionnal
            1 : plots a graphic of the norm of the coefficients at each iteration

        returns
        -------
        self
        """
        
        step = hinge_step
        if self.loss=='hinge':
            K = np.dot(np.diag(y), K)
        elif self.loss=='least-square':
            self.step = least_square_step

        (n_samples, n_features) = K.shape
        n_kernels = n_features/n_samples # We assume each kernel is a square matrix

        B_0 = B_1 = np.zeros(n_features) # coefficients to compute
        tol = 10**(-5)
        Z = B_1 # a linear combination of the coefficients of the 2 last iterations
        tau_1 = 1

        if mu==None:
            mu = 1/norm(np.dot(K, K.transpose()), 2)

        if self.penalty=='l11':
            prox = lambda(u):prox_l11(u, self.lambda_*mu)
        elif self.penalty=='l22':
            prox = lambda(u):prox_l22(u, self.lambda_*mu)
        elif self.penalty=='l21':
            prox = lambda(u):prox_l21(u, self.lambda_*mu, n_samples, n_kernels)
        elif self.penalty=='l12':
            prox = lambda(u):prox_l12(u, self.lambda_*mu, n_samples, n_kernels)

        if verbose==1:
            self.iter_coefs = list()
            self.iter_errors = list()

        for i in range(self.n_iter):
            B_0 = B_1 # B_(k-1) = B_(k)
            tau_0 = tau_1 #tau_(k+1) = tau_k
            B_1 = prox(Z + mu*step(y, K, Z))
            tau_1 = (1 + sqrt(1 + 4*tau_0**2))/2
            Z = B_1 + (tau_0 - 1)/tau_1*(B_1 - B_0)
            
            # Compute the error : use max in case norm(B_1)==0
            error = norm(B_1 - B_0, 2)/max(norm(B_1,2), 0.000001)
            # for test purpose : verbosity
            if verbose==1:
                self.iter_coefs.append(norm(B_1, 2))
                self.iter_errors.append(error)
                sys.stderr.write("Iteration : %d\r" % i )
                # print "iteration %d" % i

            # basic test of convergence
            if error <= tol and i>10:
                print "convergence at iteration : %d" % i
                break

        if verbose==1:
            #print "Norm of the coefficients at each iteration : %s"\
            #        % self.iter_coefs
            pass
        else:
            self.iter_coefs = None
        
        self.coefs = B_1
        return self

    def predict(self, K):
        """
        returns the prediction associated to the Kernels represented by K

        parameters
        ----------
        K : ndarray of size (n_samples, n_kernels*n_samples)

        returns
        -------
        ndarray : the prediction associated to K
        """
        if self.loss=='hinge':
            return np.sign(np.dot(K, self.coefs))
        else:
            return np.dot(K, self.coefs)
    
    def prediction_score(self, K, y):
        """ TODO : remove this method
        """
        if self.loss=='hinge':
            return np.sum(np.equal(self.predict(K), y))*100./len(y)
        else:
            print "Score not yet implemented for regression\n"
     

    def score(self, K, y, file_name=None):
        """
        Parameters
        ----------
        K : 2D numpy array
            matrix of observations

        y : numpy array
            the labels correspondings to K

        Returns
        -------
        The percentage of good classification for K
        """
        if self.loss=='hinge':
            if file_name is not None:
                self.save(K, y, file_name)
            return np.sum(np.equal(self.predict(K), y))*100./len(y)
        else:
            print "Score not yet implemented for regression\n"


    def save(self, K, y, file_name):
        """
        Saves the information contained in the class in the file_name file
        
        parameters
        ----------
        K : ndarray

        y : labels associated to K

        file_name : string
            name of the file in witch save the data
        """
        score = self.prediction_score(K, y)
        dic = Bunch()
        dic['penalty'] = self.penalty
        dic['loss'] = self.loss
        dic['score'] = score
        dic['n_iter'] = self.n_iter
        dic['lambda'] = self.lambda_
        dic['coefs'] = self.coefs
        if self.iter_coefs is not None:
            dic['iter_coefs'] = self.iter_coefs
            dic['iter_errors'] = self.iter_errors
        np.save(file_name, dic)
