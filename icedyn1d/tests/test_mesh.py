#!/usr/bin/env python

import os
import unittest
from mock import patch, MagicMock, call, DEFAULT
import numpy as np
from matplotlib import pyplot as plt

from icedyn1d.mesh import Mesh

class MeshTest(IceDyn1dTestBase):
    ''' Class for testing Mesh '''

    def test_get_interp_weights_1d(self):
        xn = np.array([1,2,3,4,5,6,7])
        xo = np.array([1,2,2.5,3,4.2,5.1,7])
        mesh = Mesh(xo)
        
        # calc weights should sum to 1
        w_calc = mesh.get_interp_weights_1d(xn)
        chk_sum = np.sum(w_calc, axis=0)
        self.assert_arrays_equal(chk_sum, np.ones_like(chk_sum))

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
        self.assert_arrays_equal(w_calc, w_true)

if __name__ == "__main__":
    unittest.main()
