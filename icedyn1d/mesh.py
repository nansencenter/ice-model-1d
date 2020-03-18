#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict

# set hmin and hmax for mesh as:
# hmin = (1 - _DEV_LIM)*hmean
# hmax = (1 + _DEV_LIM)*hmean
_DEV_LIM = .5

# set min size for a group of elements to be split as hmin = _HMIN_FACTOR*hmean
_HMIN_FACTOR = 2

class Mesh:

    def __init__(self, x):
        ids = np.arange(len(x), dtype=int)
        self.update_nodes(x, ids)
        self.hmin = (1 - _DEV_LIM)*self.hmean
        self.hmax = (1 + _DEV_LIM)*self.hmean

    def update_nodes(self, x, ids):
        self.nodes_x = np.array(x) #copy
        self.num_nodes = len(x)
        self.num_elements = self.num_nodes - 1
        self.nodes_id = np.array(ids, dtype=int)

    @property
    def hmean(self):
        return np.mean(self.get_widths())

    @property
    def min_split_width(self):
        return _HMIN_FACTOR*self.hmean

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

    @staticmethod
    def get_regrid_weights(shp, weights):
        w = defaultdict(lambda : np.zeros(shp))
        for inds_new, inds_old, w_cav in weights:
            for k, v in w_cav.items():
                w[k][inds_new[0]:inds_new[-1]+1, inds_old[0]:inds_old[-1]+1] = np.array(v)
        return w

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

    def extend_small_cavities_iter(self, remesh, widths):
        '''
        if an element/group of elements is too small,
        we need to merge with (a) neighbour(s)
        
        Parameters:
        -----------
        remesh: np.ndarray(bool)
            True if elements need to be remeshed
        widths: np.ndarray(float)
            element widths

        Returns:
        --------
        remesh : np.ndarray(bool)
            Modified input
        stop : bool
            repeat call again as some cavities are still too small
        '''
        stop = True
        for inds, remesh_ in self.split_cavities(remesh):
            if not remesh_:
                continue
            inds_ = list(inds)
            htot = np.sum(widths[inds_])
            both = False
            if htot < self.min_split_width:
                i0 = inds_[0] - 1
                i1 = inds_[-1] + 1
                if inds[0] == 0:
                    # can only extend to right
                    left = False
                elif inds[-1] == self.num_elements-1:
                    # can only extend to left
                    left = True
                else:

                    h0 = htot + widths[i0]
                    h1 = htot + widths[i1]
                    both = (h0 == h1)
                    if h0>=self.min_split_width and h1<self.min_split_width:
                        left = True
                    elif h1>=self.min_split_width and h0<self.min_split_width:
                        left = False
                    elif h0>=self.min_split_width and h1>=self.min_split_width:
                        # both big enough
                        # - choose the one that gives the min htot (closest to self.min_split_width)
                        left = (h0<h1)
                    else:
                        # both big enough
                        # - choose the one that gives the max htot (closest to self.min_split_width)
                        left = (h0>h1)

                if both:
                    inds_ = [i0, *inds_, i1]
                elif left:
                    inds_ = [i0, *inds_]
                else:
                    inds_ = [*inds_, i1]
                htot = np.sum(widths[inds_])
            remesh[np.array(inds_, dtype=int)] = True
            stop = stop and (htot>=self.min_split_width)
        return remesh, stop

    def extend_small_cavities(self, remesh, widths):
        '''
        if an element/group of elements is too small,
        we need to merge with (a) neighbour(s)
        
        Parameters:
        -----------
        remesh: np.ndarray(bool)
            True if elements need to be remeshed
        widths: np.ndarray(float)
            element widths

        Returns:
        --------
        remesh : np.ndarray(bool)
            Modified input
        '''
        stop = False
        while not stop:
            remesh, stop = self.extend_small_cavities_iter(remesh, widths)
        return remesh

    def detect_cavities(self):
        '''
        check for flipped or too-deformed elements

        Returns:
        --------
        remesh: np.ndarray(bool)
            True if elements need to be remeshed
        widths: np.ndarray(float)
            element widths
        '''
        # 1st check for flipping, or too small/too large elements
        widths = self.get_widths()
        remesh = (self.detect_flipped_cavities()
                + (widths<self.hmin) + (widths>self.hmax))
        # make sure none of the cavities are too small to split
        return self.extend_small_cavities(remesh, widths), widths

    def split_cavity(self, cav_widths, nodes_x_cav, next_id):
        htot = np.sum(cav_widths)
        nel_old = len(cav_widths)
        xl = nodes_x_cav[0]
        el_x_cav = .5*(nodes_x_cav[1:] + nodes_x_cav[:-1])
        nel_new = int(htot/self.hmean) # > _HMIN_FACTOR by definition
        h = htot/nel_new
        w_cons = np.array(nel_new*[cav_widths/htot])
        w_near = []
        xe = xl - h/2
        for n in range(nel_new):
            xe += h
            w_near += [np.zeros(nel_old)]
            i_near = np.argsort(np.abs(xe - el_x_cav))[0]
            w_near[-1][i_near] = 1
        id_new = np.arange(1, nel_new, dtype=int)
        x_new = xl + h*id_new.astype(float)
        id_new += next_id
        return (list(x_new), list(id_new),
                dict(conservative=w_cons, nearest=w_near))

    def adapt_mesh(self, remesh, widths):
        nodes_x = []
        nodes_id = []
        weights = []
        nel_new = 0
        next_id = self.nodes_id[-1] + 1
        for inds_old, cavity in self.split_cavities(remesh):
            n_inc = np.zeros((self.num_nodes,), dtype=bool)
            n_inc[:-1][inds_old] = True
            n_inc[1:][inds_old] = True
            if not cavity:
                nodes_x += list(self.nodes_x[n_inc])
                nodes_id += list(self.nodes_id[n_inc])
                inds_new = np.arange(
                    nel_new, nel_new + len(inds_old), dtype=int)
                nel_new += len(inds_old)
                ident = np.identity(len(inds_old))
                weights += [(inds_new, inds_old,
                    dict(conservative=ident, nearest=ident))]
            else:
                x_new, id_new, cav_weights = self.split_cavity(
                        widths[inds_old], self.nodes_x[n_inc], next_id)
                nel_add = len(x_new) + 1 #only get internal nodes from split_cavity
                nodes_x += x_new
                nodes_id += id_new
                next_id = id_new[-1] + 1
                inds_new = np.arange(
                    nel_new, nel_new + nel_add, dtype=int)
                nel_new += nel_add
                weights += [(inds_new, inds_old, cav_weights)]
        # make sure we keep the boundaries
        xl = self.nodes_x[0]
        xr = self.nodes_x[-1]
        if xl not in nodes_x:
            nodes_x.insert(0, xl)
            nodes_id.insert(0, self.nodes_x[0])
        if xr not in nodes_x:
            nodes_x.append(xr)
            nodes_id.append(self.nodes_id[-1])
        return np.array(nodes_x), np.array(nodes_id, dtype=int), weights

    def remesh(self):
        # detect cavities
        remesh, widths = self.detect_cavities()
        if not np.any(remesh):
            return
        nodes_x, nodes_id, weights_sep = self.adapt_mesh(remesh, widths)
        # update mesh and return weights for regridding tracers
        self.update_nodes(nodes_x, nodes_id)
        shp = (self.num_elements, len(widths)) #shape of full weight matrix
        weights = self.get_regrid_weights(shp, weights_sep)
