#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
#from skimage.morphology import binary_closing

class Mesh:

    def __init__(self, x):
        self.nodes_x = np.array(x)
        self.num_nodes = len(x)
        self.num_elements = self.num_nodes - 1

        # try to stay relatively close to original widths
        widths = self.get_widths()
        self.hmin = .5*widths
        self.hmax = 1.5*widths

    @staticmethod
    def vector_min(x, y):
        return .5*(x+y -np.abs(x-y))

    @staticmethod
    def vector_max(x, y):
        return .5*(x+y +np.abs(x-y))

    @staticmethod
    def split_cavities(barray):
        '''
        take a bool array and split into groups of contiguous elements

        Parameters:
        -----------
        barray : np.ndarray(bool)

        Returns:
        --------
        labeled_segments : list
            [(inds0, label0), (inds1, label1), ..., (inds_n, label_n)];
            inds0,... are np.ndarray(int) containing the indices of each segment;
            label0,... are bool, with the value coming from their value in barray
        '''
        inds = np.arange(len(barray),dtype=int)
        diff = np.diff(barray.astype(int))
        label = barray[0]
        lh = dict()
        rh = dict()
        for k, v in zip([True, False], [1, -1]):
            lh[k] = list(inds[1:][diff==v])
            rh[k] = list(inds[:-1][diff==-v])
        lh[label].insert(0, 0)
        rh[barray[-1]] += [len(barray)-1]

        out = []
        while True:
            if len(lh[label])==0:
                break
            l = lh[label].pop(0)
            r = rh[label].pop(0)
            out += [(inds[l:r+1], label)]
            label = not label
        return out

    def get_interp_weights_conservative(self, xn):
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
            w[check, i] = (self.vector_min(xrn[check], xro)
                          - self.vector_max(xln[check], xlo)
                         )/(xro - xlo)
        return w

    def move(self, um):
        '''
        Move the mesh nodes by a given displacement vector

        Parameters:
        -----------
        um : numpy.ndarray
        '''
        self.nodes_x += um

    def get_jacobian(self):
        '''
        return the jacobian, jac, of the transformation between the reference element , [0,1],
        and the mesh elements.
        The transformation is x = x0 + (x1-x0)*xi
        jac < 0 means the element has flipped

        Returns:
        --------
        jac : numpy.ndarray
        '''
        return np.diff(self.nodes_x)

    def get_widths(self):
        '''
        return the element widths

        Returns:
        --------
        w : numpy.ndarray
        '''
        return np.abs(self.get_jacobian())

    def get_mass_matrix(self):
        '''
        M_{i,j} = \int_{x_0}^{x_1} { N_i(x)N_j(x) } dx

        Returns:
        --------
        M : list
            [[M_00, M_01], [M_01, M_11]]
        '''
        jac = self.get_jacobian()
        return [[jac/3, jac/6], [jac/6, jac/3]]

    def get_shape_coeffs(self):
        '''
        get the shape coefficients - the gradients of the basis functions for each element
        N_0(xi) = xi, N_1(xi) = 1-xi;
        \pa_x N_0(x) = (dxi/dx)*\pa_xi N_0(xi) = 1/J
        \pa_x N_1(x) = (dxi/dx)*\pa_xi N_1(xi) = -1/J
        '''
        jac = self.get_jacobian()
        return [1/jac, -1/jac]

    def get_bad_elements(self):
        '''
        check for flipped or too-deformed elements

        Returns:
        --------
        bad_elements : numpy.ndarray(bool)
        '''
        jac = self.get_jacobian()
        widths = np.abs(jac)
        bad_elements = (jac<0) + (widths<self.hmin) + (widths>self.hmax)
        return bad_elements, widths

    def detect_flipped_cavities(self):
        '''
        If flipping has occurred, label elements that need to be remeshed
        
        Returns:
        --------
        remesh: np.ndarray(bool)
            True if elements need to be remeshed
        '''
        jac = self.get_jacobian()
        lh = self.nodes_x[:-1]
        rh = self.nodes_x[1:]
        remesh = np.zeros_like(lh, dtype=bool)
        for inds, flipped in self.split_cavities(jac<0):
            if not flipped:
                continue
            x_l = lh[inds[0]]
            x_r = rh[inds[-1]]
            remesh += (rh>=x_r)*(lh<=x_l)
        return remesh

    def remesh(self):
        cavities, widths = self.detect_cavities()
