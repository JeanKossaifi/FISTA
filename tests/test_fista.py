from __future__ import division

__author__ = 'Jean KOSSAIFI'

import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_almost_equal

from ..fista import prox_l11, prox_l22, prox_l21, prox_l12, compute_M, mixed_norm, by_kernel_norm
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

def test_prox_l12():
    # test of the shrinkage
    l = 0
    u = np.ones(8)
    assert_array_equal(u, prox_l12(u, l, 4, 2))

    l = 1
    # ! u must be of dtype np.float
    u = np.array([0, 3, 4, 0], dtype=np.float)
    res = u - 7/15
    res[0] = res[3] = 0
    assert_array_equal(res, prox_l12(u, l, 4, 1))

def test_compute_M():
    l = 1
    u = np.arange(4)
    Ml, sum_Ml = compute_M(u, l, len(u))
    assert_true(Ml==2)
    assert_true(sum_Ml==5)

def test_mixed_norm():
    u = np.ones(8)
    n_kernels, n_samples = 2, 4
    # Test of l_12 norm
    assert_almost_equal(mixed_norm(u, 1, 2, n_samples, n_kernels), 5.65685424)
    # Test of l_12 norm
    assert mixed_norm(u, 2, 1, n_samples, n_kernels) == 4
    # Test of l_22 norm
    assert_almost_equal(mixed_norm(u, 2, 2, n_samples, n_kernels),
            2.828427124746)
    # Test of l_11 norm
    assert mixed_norm(u, 1, 1, n_samples, n_kernels) == 8

def test_by_kernel_norm():
    u = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert by_kernel_norm(u, 2, 2, 4, 2) == [2, 0]

def test_Fista():
    generator = np.random.RandomState(seed=0)
    fista = Fista(lambda_=0.5, loss='squared-hinge', penalty='l11', n_iter=1000)
    X = generator.normal(size=(10, 40))
    y = np.sign(generator.normal(size=10))
    # Test for norm l11
    fista.fit(X, y)
    assert fista.score(X, y) == 100
    # Checking sparcity
    assert len(fista.coefs_[fista.coefs_==0]) > 1

    # Test for norm l12
    fista.penalty='l21'
    fista.fit(X, y)
    assert fista.score(X, y) == 100
    # Checking sparcity
    # The norm should nul entire kernels
    assert len(fista.coefs_[fista.coefs_==0]) % 10 == 0
    
    fista.penalty='l12'
    fista.fit(X, y)
    assert fista.score(X, y) == 100
    # Checking sparcity
    coefs_ = fista.coefs_.reshape(4, 10)
    # For every kernel, some samples should be nulled
    # But not all of them
    for i in coefs_:
        assert len(i[i==0]) > 0
        assert len(i[i!=0]) > 0

    fista.penalty='l22'
    fista.fit(X, y, verbose=1)
    assert fista.score(X, y) == 100
