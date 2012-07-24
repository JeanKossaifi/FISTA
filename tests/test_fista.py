__author__ = 'Jean KOSSAIFI'
__license__ = 'BSD'

import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_true

from ..fista import prox_l11, prox_l22, prox_l21, compute_M, norm_l12, norm_l21
from ..fista import Fista

def test_prox_l11():
    u = np.ones(3)
    l = 0.
    assert_array_equal(u, prox_l11(u, l))
    l = 0.5
    assert_array_equal(0.5*np.ones(3), prox_l11(u, l))
    l = 1
    assert_array_equal(np.zeros(3), prox_l11(u, l))
    u = -1 * u
    l = 0.5
    assert_array_equal(-0.5*np.ones(3), prox_l11(u, l))
    
def test_prox_l22():
    l = 1
    u = np.ones(3)
    assert_array_equal(np.array([0.5, 0.5, 0.5]), prox_l22(u, l))

def test_prox_l21():
    l = 2
    u = np.ones(8)
    assert_array_equal(np.zeros(8), prox_l21(u, l, 4, 2))
    l = 1
    assert_array_equal(u*0.5, prox_l21(u, l, 4, 2))
    u = np.array([1., 1., 1., 1., 0., 0., 2., 0.])
    assert_array_equal([0.5, 0.5, 0.5, 0.5, 0, 0, 1, 0], prox_l21(u, l, 4, 2))

def test_compute_M():
    l = 1
    u = np.arange(4)
    Ml, sum_Ml = compute_M(u, l, len(u))
    assert_true(Ml==2)
    assert_true(sum_Ml==5)

def test_norm_l12():
    u = np.ones(8)
    n_kernels, n_samples = 4, 2
    assert norm_l12(u, n_samples, n_kernels) == 4

def test_norm_l21():
    u = np.ones(8)
    n_kernels, n_samples = 2, 4
    assert norm_l21(u, n_samples, n_kernels) == 4

def test_Fista():
    fista = Fista(lambda_=0.5, loss='hinge', penalty='l11', n_iter=1000)
    X = np.random.normal(size=(10, 40))
    y = np.sign(np.random.normal(size=10))
    fista.fit(X, y)
    assert fista.prediction_score(X, y) == 1
    fista.penalty='l12'
    fista.fit(X, y)
    assert fista.prediction_score(X, y) == 1
    fista.penalty='l21'
    fista.fit(X, y)
    assert fista.prediction_score(X, y) == 1
    fista.penalty='l22'
    fista.fit(X, y, verbose=1)
    assert fista.prediction_score(X, y) == 1
