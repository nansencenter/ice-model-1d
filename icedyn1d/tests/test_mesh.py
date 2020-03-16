#!/usr/bin/env python

import numpy as np
import unittest
from mock import patch, MagicMock, call, DEFAULT

from icedyn1d.mesh import Mesh
from icedyn1d.tests.icedyn1d_test_base import IceDyn1dTestBase

class MeshTest(IceDyn1dTestBase):
    ''' Class for testing Mesh '''

    def test_get_interp_weights_conservative(self):
        xn = np.array([1,2,3,4,5,6,7])
        xo = np.array([1,2,2.5,3,4.2,5.1,7])
        mesh = Mesh(xo)
        
        # calc weights should sum to 1
        w_calc = mesh.get_interp_weights_conservative(xn)
        chk_sum = np.sum(w_calc, axis=0)
        self.assert_arrays_equal(chk_sum, np.ones_like(chk_sum), tol=1e-8)

        #true weights:
        #rows are for new elements
        #columns are weights for old elements
        w_true = np.zeros((6,6))
        w_true[0,0] = 1
        w_true[1,1:3]=1
        w_true[2,3]=1/1.2
        w_true[3,3]=.2/1.2
        w_true[3,4]=.8/.9
        w_true[4,4]=.1/.9
        w_true[4,5]=.9/1.9
        w_true[5,5]=1/1.9
        self.assert_arrays_equal(w_calc, w_true, tol=1e-8)

    def test_split_cavities(self):
        # eg of only internal cavities
        b= np.array([0,1,0,1,1,1,0,0,0], dtype=bool)
        segs = Mesh.split_cavities(b)
        segs2 = [
                (np.array([0]), False),
                (np.array([1]), True),
                (np.array([2]), False),
                (np.array([3, 4, 5]), True),
                (np.array([6, 7, 8]), False),
                ]
        self.assertEqual(len(segs), len(segs2))
        for il, il2 in zip(segs, segs2):
            i, l = il
            i2, l2 = il2
            self.assert_arrays_equal(i, i2)
            self.assertEqual(l, l2)

        # eg where there is an end cavity
        b= np.array([0,1,0,1,1,1,0,0,1], dtype=bool)
        segs = Mesh.split_cavities(b)
        segs2 = [
                (np.array([0]), False),
                (np.array([1]), True),
                (np.array([2]), False),
                (np.array([3, 4, 5]), True),
                (np.array([6, 7]), False),
                (np.array([8]), True),
                ]
        self.assertEqual(len(segs), len(segs2))
        for il, il2 in zip(segs, segs2):
            i, l = il
            i2, l2 = il2
            self.assert_arrays_equal(i, i2)
            self.assertEqual(l, l2)

    def test_vector_min(self):
        # eg of only internal cavities
        x = np.arange(5, dtype=float)
        y = Mesh.vector_min(x, 2.5)
        x[x>2.5] = 2.5
        self.assert_arrays_equal(y, x)

    def test_vector_max(self):
        # eg of only internal cavities
        x = np.arange(5, dtype=float)
        y = Mesh.vector_max(x, 2.5)
        x[x<2.5] = 2.5
        self.assert_arrays_equal(y, x)

if __name__ == "__main__":
    unittest.main()
