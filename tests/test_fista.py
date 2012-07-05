__author__ = 'Jean KOSSAIFI'
__license__ = 'BSD'

import numpy as np

from numpy.testing import assert_array_equal
from nose.tools import assert_true

from ..fista import prox_l11, prox_l22, prox_l21, compute_M

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

def test_compute_M():
    l = 1
    u = np.arange(4)
    Ml, sum_Ml = compute_M(u, l, len(u))
    assert_true(Ml==2)
    assert_true(sum_Ml==5)
