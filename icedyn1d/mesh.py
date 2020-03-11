#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt


class mesh:

    def __init__(self, x):
        self.nodes_x = x
        self.num_nodes = len(x)
        self.num_elements = self.num_nodes - 1

    def vector_min(self, x, y):
        return .5*(x+y -np.abs(x-y))

    def vector_max(self, x, y):
        return .5*(x+y +np.abs(x-y))

    def get_interp_weights_1d_conservative(self, xn):
        '''
        1d conservative remapping routine

        Parameters:
        -----------
        xn : numpy.ndarray
            nodes of new mesh

        Returns:
        -----------
        w : numpy.ndarray
            matrix to convert field on mesh elements to
            another mesh
        '''
        n_el_new = len(xn) -1
        w = np.zeros((n_el_new, self.num_elements))
        xln = xn[:-1] #new LH nodes
        xrn = xn[1:]  #new RH nodes
        for i in range(self.num_elements):
            # loop over old nodes
            xlo, xro = self.nodes_x[i:i+2] #old LH, RH nodes
            check = (xrn>xlo) * (xln<xro)
            w[check, i] = (vector_min(xrn[check], xro)
                          - vector_max(xln[check], xlo)
                         )/(xro - xlo)
        return w
