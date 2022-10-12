import os 
import copy
import logging
import pyopencl as cl
import numpy as np
from numpy.core.arrayprint import dtype_is_implied
import numpy.fft as fft
import matplotlib.pyplot as plt

from scipy.signal import butter, hilbert, freqz
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter

from PyFWI import rock_physics as rp


def derivative(nx, nz, dx, dz, order):
    """
    Compute spatial derivative operators for grid _cells_
    For 1st order: \n
    \tforward operator is (u_{i+1} - u_i)/dx\n
    \tcentered operator is (u_{i+1} - u_{i-1})/(2dx)\n
    \tbackward operator is (u_i - u_{i-1})/dx \n

    For 2nd order: \n
    \tforward operator is (u_i - 2u_{i+1} + u_{i+2})/dx^2 \n
    \tcentered operator is (u_{i-1} - 2u_i + u_{i+1})/dx^2 \n
    \tbackward operator is (u_{i-2} - 2u_{i-1} + u_i)/dx^2 \n

    Parameters
    ----------
        nx : int
            Number of samples in X-direction
        nz : int
            Number of samples in Z-direction
        dx : float
            Samplikng rate in X-direction
        dz : float
            Samplikng rate in Z-direction
        order : int
            Order of derivative

    Returns
    -------
        Dx : Dispersed matrix
            Derivative matrix in X-direction
        Dz : Dispersed matrix
            Derivative matrix in Z-direction

    """

    if order == 1:

        # forward operator is (u_{i+1} - u_i)/dx
        # centered operator is (u_{i+1} - u_{i-1})/(2dx)
        # backward operator is (u_i - u_{i-1})/dx

        idx = 1 / dx
        idz = 1 / dz

        i = np.kron(np.arange(nz * nx), np.ones((2,), dtype=np.int64))
        j = np.zeros((nz * nx * 2,), dtype=np.int64)
        v = np.zeros((nz * nx * 2,))

        jj = np.vstack((np.arange(nx), nx + np.arange(nx))).T
        jj = jj.flatten()
        j[:2 * nx] = jj

        vd = idx * np.tile(np.array([-1, 1]), (nx,))
        v[:2 * nx] = vd

        jj = np.vstack((-nx + np.arange(nx), nx + np.arange(nx))).T
        jj = jj.flatten()
        for n in range(1, nz - 1):
            j[n * 2 * nx:(n + 1) * 2 * nx] = n * nx + jj
            v[n * 2 * nx:(n + 1) * 2 * nx] = 0.5 * vd

        jj = np.vstack((-nx + np.arange(nx), np.arange(nx))).T
        jj = jj.flatten()
        j[(nz - 1) * 2 * nx:nz * 2 * nx] = (nz - 1) * nx + jj
        v[(nz - 1) * 2 * nx:nz * 2 * nx] = vd

        Dz = sp.csr_matrix((v, (i, j)))

        jj = np.vstack((np.hstack((0, np.arange(nx - 1))),
                        np.hstack((np.arange(1, nx), nx - 1)))).T
        jj = jj.flatten()
        vd = idz * np.hstack((np.array([-1, 1]),
                              np.tile(np.array([-0.5, 0.5]), (nx - 2,)),
                              np.array([-1, 1])))

        for n in range(nz):
            j[n * 2 * nx:(n + 1) * 2 * nx] = n * nx + jj
            v[n * 2 * nx:(n + 1) * 2 * nx] = vd

        Dx = sp.csr_matrix((v, (i, j)))
    else:  # 2nd order
        idx2 = 1 / (dx * dx)
        idz2 = 1 / (dz * dz)

        i = np.kron(np.arange(nz * nx), np.ones((3,), dtype=np.int64))
        j = np.zeros((nz * nx * 3,), dtype=np.int64)
        v = np.zeros((nz * nx * 3,))

        jj = np.vstack((np.arange(nx), nx + np.arange(nx),
                        2 * nx + np.arange(nx))).T
        jj = jj.flatten()
        j[:3 * nx] = jj
        vd = idx2 * np.tile(np.array([1.0, -2.0, 1.0]), (nx,))
        v[:3 * nx] = vd

        for n in range(1, nz - 1):
            j[n * 3 * nx:(n + 1) * 3 * nx] = (n - 1) * nx + jj
            v[n * 3 * nx:(n + 1) * 3 * nx] = vd

        j[(nz - 1) * 3 * nx:nz * 3 * nx] = (nz - 3) * nx + jj
        v[(nz - 1) * 3 * nx:nz * 3 * nx] = vd

        Dz = sp.csr_matrix((v, (i, j)))

        jj = np.vstack((np.hstack((0, np.arange(nx - 2), nx - 3)),
                        np.hstack((1, np.arange(1, nx - 1), nx - 2)),
                        np.hstack((2, np.arange(2, nx), nx - 1)))).T
        jj = jj.flatten()
        vd = vd * idz2 / idx2

        for n in range(nz):
            j[n * 3 * nx:(n + 1) * 3 * nx] = n * nx + jj
            v[n * 3 * nx:(n + 1) * 3 * nx] = vd

        Dx = sp.csr_matrix((v, (i, j)))

    return Dx, Dz


class Regularization:
    """
    Regularization Prepares tools for regularizing FWI problem

    Parameters
    ----------
    nx : int scalar
        Number of samples in x-direction
    nz : int scalar
        Number of samples in z-direction
    dx : float scalar
        Spatial sampling rate in x-direction
    dz : float scalar
        Spatial sampling rate in z-direction
    """
    def __init__(self, nx, nz, dx, dz):

        self.idx = 1 / dx
        self.idz = 1 / dz

        self.idx2 = 1 / (dx * dx)
        self.idz2 = 1 / (dz * dz)

        self.dx = dx
        self.dz = dz

        self.nx = nx
        self.nz = nz
        self.n_elements = nz * nx

        self.Bx2, self.Bz2 = derivative(nx, nz, dx, dz, 2)
        self.Bx1, self.Bz1 = derivative(nx, nz, dx, dz, 1)

        self.D2 = self.Bx2.T @ self.Bx2 + self.Bz2.T @ self.Bz2

    def cost_regularization(self, x0,
                            tv_properties=None,
                            tikhonov_properties=None,
                            tikhonov0_properties=None):
        x = np.copy(x0)
        rms = 0
        grad = np.zeros(x.shape)

        # Because we may provide the properties but ask for regularization in some special frequencies
        if tv_properties:
            f_tv, g_tv = self.tv(x, 1e-7, alpha_z=tv_properties['az'], alpha_x=tv_properties['ax'])

            rms += tv_properties['lambda_weight'] * f_tv
            grad += tv_properties['lambda_weight'] * g_tv

        if tikhonov_properties:
            f_tikh, g_tikh = self.tikhonov(x, alpha_z=tikhonov_properties['az'], alpha_x=tikhonov_properties['ax'])

            rms += tikhonov_properties['lambda_weight'] * f_tikh
            grad += tikhonov_properties['lambda_weight'] * g_tikh

        if tikhonov0_properties:
            f_tikh, g_tikh = self.tikhonov_0(x)

            rms += tikhonov0_properties['lambda_weight'] * f_tikh
            grad += tikhonov0_properties['lambda_weight']  # No gradient

        return rms, grad

    def tv(self, x0, eps, alpha_z, alpha_x):
        """
        Parameters
        ----------
        x0 : float
            Data
        eps : scalar float
            small value for make it deffrintiable at zero
        alpha_z : scalar float
            coefficient of Dz
        alpha_x : scalar float
            coefficient of Dx
            
        Returns
        -------
        rms : scalar float
            loss
        grad : scalar float
            Gradient of loss w.r.t. model parameters
        """
        x = np.copy(x0)

        ln = (self.nx*self.nz)
        ln_x = len(x)
        n = ln_x//ln  # NOT self.n_parameter

        x1 = np.zeros(ln_x,)
        for i in range(n):
            mx1 = self.Bx1 @ x[i*ln:(i+1)*ln]
            mz1 = self.Bz1 @ x[i*ln:(i+1)*ln]

            # To ignore the effect of sharp change after first 15 samples
            mz1[:17*self.nx] = 0.0

            x1[i*ln:(i+1)*ln] = alpha_x * mx1 + alpha_z * mz1
        rms, grad = self.l1(x1, eps)

        return rms, grad

    @staticmethod
    def l1(x0, eps=1e-7):
        x = np.copy(x0)

        x1 = np.copy(x)
        len_x = len(x)
        x1[np.abs(x1) <= eps] = eps
        w_1 = np.sqrt(np.abs(x1))
        w = sp.spdiags(1/w_1, diags=0, m=len_x,  n=len_x)

        wx = w@x
        rms = wx.T @ wx  # np.sum(np.abs(x))

        grad = 2 * wx.T @ w

        return rms, grad

    @staticmethod
    def l2(x0):
        x = np.copy(x0)
        rms = x.T @ x

        return rms

    def tikhonov(self, x0, alpha_z, alpha_x):
        """
        A method to implement Tikhonov regularization with order of 2

        Parameters
        ----------
            x0 : 1D ndarray
                Data
            alpha_z : float
                coefficient of Dz
            alpha_x : float
                coefficient of Dx

        Returns
        -------
        rms : scalar float
            loss
        grad : scalar float
            Gradient of loss w.r.t. model parameters
        """
        x = np.copy(x0)
        ln = (self.nx * self.nz)
        ln_x = len(x)
        n = ln_x // ln  # NOT self.n_parameter

        x1 = np.zeros(ln_x,)
        grad = np.zeros(x.shape)
        for i in range(n):
            m = np.copy(x[i * ln:(i + 1) * ln])
            mx1 = self.Bx1 @ m
            mz1 = self.Bz1 @ m

            # To ignore the effect of sharp change after first 15 samples
            mz1[:17 * self.nx] = 0.0

            x1[i * ln:(i + 1) * ln] = alpha_x * mx1 + alpha_z * mz1
            grad[i * ln:(i + 1) * ln] = alpha_x * (self.Bx1.T @ self.Bx1) @ m +\
                                        alpha_z * (self.Bz1.T @ self.Bz1) @ m

        rms = self.l2(x1)

        return rms, grad

    def tikhonov_0(self, x0):
        """
        A method to implement Tikhonov regularization with order of 0

        Parameters
        ----------
            x0 : 1D ndarray
                Data
            alpha_z : float
                coefficient of Dz
            alpha_x : float
                coefficient of Dx

        Returns
        -------
            rms : float
                error
            grad : 1D ndarray
                gradient of the regularization
        """
        x = np.copy(x0)

        ln = (self.nx * self.nz)
        ln_x = len(x)
        n = ln_x // ln  # NOT self.n_parameter

        x1 = np.zeros(ln_x, )
        for i in range(n):
            mx1 = x[i * ln:(i + 1) * ln]

            x1[i * ln:(i + 1) * ln] = mx1
        rms, grad = self.l1(x1)

        return rms, grad

    def parameter_relation(self, m0, models, k0, kend, freq):
        """
        parameter_relation considers regularization for the
        relation between parameters.


        Parameters
        ----------
        m0 : ndarray
            Vector of parameters
        models : dict
            A dictionary containing couple of dictionaries which includes a numpy
            polyfit model and regularization parameter.
        k0 : int
            Index of the first parameter in m0
        kend : int
            Index of the last parameter in m0


        Returns
        -------
        rms : float
            rms of regularization
        grad: ndarray
            Vector of gradient od the regularization
        """
        rms = 0
        grad = np.zeros(m0.shape)

        for param in models:
            par = [char for char in param]
            model = models[param]['model']
            lam = models[param]['lam']

            desired_freq  = models[param]['freqs']

            if freq not in np.array(desired_freq).reshape(-1):
                # has to be written like that to work eaither if freq is given as int or list
                return 0.0, grad

            par_int = np.int32(par)
            if par_int[1] in np.arange(k0+1, kend+1):
                pre21 = model(m0[(par_int[0]-1) * self.n_elements:par_int[0] * self.n_elements])

                dm21 = m0[(par_int[1]-1)  * self.n_elements:par_int[1] * self.n_elements] - pre21

                rms += 0.5 * lam * np.dot(dm21.T, dm21)
                grad[(par_int[1]-1)  * self.n_elements: par_int[1] *self.n_elements] = gaussian_filter(lam * dm21 * 1, 1)

        return rms, grad

    def priori_regularization(self, m0, regularization_dict, k0, kend, freq):
        """
        priori_regularization consider the priori information regularization.


        Parameters
        ----------
        m0 : float
            Vector of parameters
        regularization_dict : dict
            A dictionary containing couple of priori model and regularization hyperparameter
        k0 : int
            Index of the first parameter in m0
        kend : int
            Index of the last parameter in m0

        Returns
        -------
        rms : float
            rms of regularization
        grad: ndarray
            Vector of gradient od the regularization

        References
        ----------
        Asnaashari et al., 2013, Regularized seismic full waveform inversion with prior model information, Geophysics, 78(2), R25-R36, eq. 5.
        """
        if regularization_dict is None:
            return 0.0, np.zeros(m0.shape, np.float64)

        m0 = np.copy(m0[: kend * self.n_elements])
        mp = np.zeros(m0.shape)
        desired_freq  = regularization_dict['freqs']

        if freq not in np.array(desired_freq).reshape(-1):
            return 0.0, np.zeros(m0.shape, np.float64)

        lam = regularization_dict['lam']

        mp_dict = regularization_dict['mp']

        for i in range(kend - k0):
            mp[i * self.n_elements: (i + 1) * self.n_elements] = mp_dict[[*mp_dict][k0 + i]].reshape(-1)

        ii = jj = np.arange((kend - k0) * self.n_elements)
        v = np.ones((ii.shape)) / np.var(mp)

        W = sp.csr_matrix((v, (ii, jj)))

        diff = (m0 - mp).reshape(-1, 1)

        rms = lam * 0.5 * (diff.T @ W) @ diff

        grad = lam * W.T @ diff

        return rms.item(), grad.reshape(-1)




class Fdm(object):
    def __init__(self, order):
        """
        Fdm is a class to implemenet the the finite difference method for wave modeling

        The coeeficients are based on Lavendar, 1988 and Hasym et al., 2014.

        Args:
            order (int, optional): [description]. Defaults to 4.
        """
        self._order = order

        if order == 4:
            self._c1 = 9/8
            self._c2 = - 1 / 24
            self._c3 = 0
            self._c4 = 0

        elif order == 8:
            self._c1 = 1715 / 1434
            self._c2 = -114 / 1434
            self._c3 = 14 / 1434
            self._c4 = -1 / 1434

        else:
            raise AssertionError ("Order of the derivative has be either 4 or 8!")

        dh_n = { # Dablain, 1986, Bai et al., 2013
            '4': 4,
            '8': 3
        }

        self.dh_n = dh_n[str(order)] # Pick the appropriate n for calculating dh

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if value in [4, 8]:
            self._order = value
        else:
            raise AssertionError ("Order of the derivative has be either 4 or 8!")

    @property
    def c1(self):
        return self._c1

    @c1.setter
    def c1(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")

    @property
    def c2(self):
        return self._c2

    @c2.setter
    def c2(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")

    @property
    def c3(self):
        return self._c3

    @c3.setter
    def c3(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")

    @property
    def c4(self):
        return self._c4

    @c4.setter
    def c3(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")


    def dxp(self, x, dx):
         if self.order == 4:
             return self._dxp4(x, dx)
         else:
             return self._dxp8(x, dx)

    def dxm(self, x, dx):
         if self.order == 4:
             return self._dxm4(x, dx)
         else:
             return self._dxm8(x, dx)

    def dzp(self, x, dx):
         if self.order == 4:
             return self._dzp4(x, dx)
         else:
             return self._dzp8(x, dx)

    def dzm(self, x, dx):
         if self.order == 4:
             return self._dzm4(x, dx)
         else:
             return self._dzm8(x, dx)

    def _dxp4(self, x, dx):
        y = np.zeros(x.shape)

        y[2:-2, 2:-2] = (self._c1 * (x[2:-2, 3:-1] - x[2:-2, 2:-2]) +
                         self._c2 * (x[2:-2, 4:] - x[2:-2, 1:-3])) / dx
        return y

    def _dxp8(self, x, dx):
        y = np.zeros(x.shape)

        y[4:-4, 4:-4] = (self._c1 * (x[4:-4, 5:-3] - x[4:-4, 4:-4]) +
                         self._c2 * (x[4:-4, 6:-2] - x[4:-4, 3:-5]) +
                         self._c3 * (x[4:-4, 7:-1] - x[4:-4, 2:-6]) +
                         self._c4 * (x[4:-4, 8:  ] - x[4:-4, 1:-7])) / dx
        return y


    def _dxm4(self, x, dx):
        y = np.zeros((x.shape))

        y[2:-2, 2:-2] = (self._c1 * (x[2:-2, 2:-2] - x[2:-2, 1:-3]) +
                         self._c2 * (x[2:-2, 3:-1] - x[2:-2, :-4])) / dx
        return y

    def _dxm8(self, x, dx):
        y = np.zeros((x.shape))

        y[4:-4, 4:-4] = (self._c1 * (x[4:-4, 4:-4] - x[4:-4, 3:-5]) +
                         self._c2 * (x[4:-4, 5:-3] - x[4:-4, 2:-6]) +
                         self._c3 * (x[4:-4, 6:-2] - x[4:-4, 1:-7]) +
                         self._c4 * (x[4:-4, 7:-1] - x[4:-4, :-8]) ) / dx
        return y

    def _dzp4(self, x, dx):
        y = np.zeros(x.shape)

        y[2:-2, 2:-2] = (self.c1 * (x[3:-1, 2:-2] - x[2:-2, 2:-2]) +
                         self.c2 * (x[4:, 2:-2] - x[1:-3, 2:-2])) / dx
        return y

    def _dzp8(self, x, dx):
        y = np.zeros(x.shape)

        y[4:-4, 4:-4] = (self.c1 * (x[5:-3, 4:-4] - x[4:-4, 4:-4]) +
                         self.c2 * (x[6:-2, 4:-4] - x[3:-5, 4:-4]) +
                         self.c3 * (x[7:-1, 4:-4] - x[2:-6, 4:-4]) +
                         self.c4 * (x[8: , 4:-4] - x[1:-7, 4:-4])) / dx
        return y


    def _dzm4(self, x, dx):
        y = np.zeros((x.shape))

        y[2:-2, 2:-2] = (self.c1 * (x[2:-2, 2:-2] - x[1:-3, 2:-2]) +
                         self.c2 * (x[3:-1, 2:-2] - x[:-4, 2:-2])) / dx
        return y

    def _dzm8(self, x, dx):
        y = np.zeros((x.shape))

        y[4:-4, 4:-4] = (self.c1 * (x[4:-4, 4:-4] - x[3:-5, 4:-4]) +
                         self.c2 * (x[5:-3, 4:-4] - x[2:-6, 4:-4]) +
                         self.c3 * (x[6:-2, 4:-4] - x[1:-7, 4:-4]) +
                         self.c4 * (x[7:-1, 4:-4] - x[ :-8, 4:-4])) / dx
        return y


    def dot_product_test_derivatives(self):

        x = np.random.rand(100, 100)
        x[:4, :] = x[-4:, :] = x[:, :4] = x[:, -4:] = 0

        y = np.random.rand(100, 100)
        y[:4, :] = y[-4:, :] = y[:, :4] = y[:, -4:] = 0

        error_x = np.sum(x * self.dxp(y, 1)) - np.sum(- self.dxm(x, 1) * y)
        error_z = np.sum(x * self.dzp(y, 1)) - np.sum(- self.dzm(x, 1) * y)

        print("Errors for derivatives are \n {}, {}".format(error_x, error_z))

    def dt_computation(self, vp_max, dx, dz=None):
        '''
        ref: Bai et al, 2013
        '''
        if dz is None:
            dz = dx

        c_sum = np.abs(self._c1) + np.abs(self._c2) + \
            np.abs(self._c3) + np.abs(self._c4)

        a = 1/dx/dx * c_sum + 1/dz/dz * c_sum
        dt = 2 / vp_max / np.sqrt(a*(1 + 4.0))

        return dt


def _acoustic_model_preparation(model, med_type):
    keys = [*model]

    len_keys = len(keys)
    shape = model[[*model][0]].shape

    model['vs'] = np.zeros(shape, np.float32)
    model['mu'] = np.zeros(shape, np.float32)

    if 'rho' not in keys:
        model['rho'] = np.ones(shape, np.float32)
        print("Density is considered constant.")

    if keys[0] == 'lam':
        model['vp'] = rp.p_velocity().lam_mu_rho(model['lam'], model['vs'], model['rho'])

    return model


def _elastic_model_preparation(model0, med_type):
    model = model0.copy()
    keys = [*model]

    len_keys = len(keys)
    shape = model[[*model][0]].shape

    if 'vp' not in keys:
        try:
            model['vp'] = 1 * rp.p_velocity().Han(model['phi'], model['cc'])
            logging.info("P-wave velocity is estimated based on Han method")

        except:
            raise Exception ("Model has to have P-wave velocity")

    if len_keys < 3:
        raise "For Elastic case (med_type=1), vp, vs, and density have to be provided."


    return model


def disperasion_stability(vp, sdo, fn):
    """
    disperasion_stability returns the appropriate parameters for FD
    that prevent unstability and dispersion


    Parameters
    ----------
    vp : float
        P-wave velocity
    sdo : int
        Spatial order of derivation
    fn : float
        Nyquist frequency

    Returns
    -------
    dh : float
        Spatial sampling rate
    dt : float
        Temporal sampling rate
    """
    D = Fdm(order=sdo)

    dh = vp.min() / (D.dh_n * fn)
    dt = D.dt_computation(vp.max(), dh)

    return dh, dt


def modeling_model(model, med_type):

    if med_type in [0, 'acoustic']:
        model = _acoustic_model_preparation(model, med_type)

    elif med_type in [1, 'elastic']:
       model = _elastic_model_preparation(model, med_type)


    return model


class Recorder:
    def __init__(self, nt, rec_loc, ns, dh):

        self.rec_loc = np.int32(rec_loc/dh)
        self.nr = rec_loc.shape[0]

        self.vx = np.zeros((nt, ns * self.nr), dtype=np.float32)
        self.vz = np.zeros((nt, ns * self.nr), dtype=np.float32)
        self.taux = np.zeros((nt, ns * self.nr), dtype=np.float32)
        self.tauz = np.zeros((nt, ns * self.nr), dtype=np.float32)
        self.tauxz = np.zeros((nt, ns * self.nr), dtype=np.float32)

    def __call__(self, t, s, **kargs):
        for key, value in kargs.items():
            exec("self." + key + "[t, s*self.nr:(s+1)*self.nr] = value[self.rec_loc[:, 1], self.rec_loc[:, 0]]")

    def acquire(self):
        data = {
            'vx': self.vx,
            'vz': self.vz,
            'taux': self.taux,
            'tauz': self.tauz,
            'tauxz': self.tauxz
        }
        return data


def residual(d_est, d_obs):
    res = {}
    for key in d_obs:
        res[key] = d_est[key] - d_obs[key]
    return res


def cost_function(d_est, d_obs):

    res = [d_est[key] - d_obs[key] for key in d_obs]
    res = np.array(res).reshape(-1, 1)

    rms = 0.5 * np.dot(res.T, res)
    return np.squeeze(rms)


def expand_model(parameter, TNz, TNx, n_pml=10):
    """
    This function make room around the 'parameter' to stick the pml layer.

    Parameters
    ----------
        parameter : float
            Matrix of property that we are going to consider pml around.

        TNz : int
            Number of samples in z-direction (`n_z + 2 * n_pml`).

        TNx : int
            Number of samples in x-direction (`n_x + 2 * n_pml`).

        n_pml : int, optional = 10
            Number of pml layer

    Returns
    --------
        nu : float
            A matrix with the size of [TNz, TNx] with zero value
            everywhere excpet in center which consisting the model.

    """

    nu = np.zeros((TNz, TNx)).astype(np.float32, order='C')
    nu[n_pml:TNz - n_pml, n_pml:TNx - n_pml] = \
        parameter.astype(np.float32, order='C')

    nu[:n_pml, :] = nu[n_pml, :]
    nu[TNz - n_pml:, :] = nu[TNz - n_pml - 1, :]

    nu[:, TNx - n_pml:] = nu[:, TNx - n_pml - 1].reshape(-1, 1)
    nu[:, :n_pml] = nu[:, n_pml].reshape(-1, 1)
    return nu


class CPML:
    def __init__(self, dh, dt, N, nd=2.0, Rc=0.001, nu0=2.0, nnu=2.0,
                 alpha0=20 * np.pi, nalpha=1.0):
        """
        Input
            N      : nombre de couches PML
            nd     : ordre du profile du facteur d'amortissement
            Rc     : coefficient de réflexion théorique à la limite des PML
            nu0    : valeur max du paramètre nu
            nnu    : ordre du profile du paramètre nu
            nalpha : ordre du profile du paramètre alpha
            alpha0 : valeur max du paramètre alpha
        """
        self.dh = dh
        self.dt = dt
        self.Npml = N
        self.nd = np.float32(nd)
        self.Rc = np.float32(Rc)
        self.nu0 = np.float32(nu0)
        self.nnu = np.float32(nnu)
        self.alpha0 = np.float32(alpha0)
        self.nalpha = np.float32(nalpha)

    def pml_prepare(self, V):
        v_max = V.max()
        [TNz, TNx] = V.shape
        nx = TNx - self.Npml
        nz = TNz - self.Npml

        zp1 = np.repeat(self.dh * np.arange(self.Npml + 1, 1, -1), TNx).reshape(self.Npml, TNx)
        zp = np.zeros((TNz, TNx), np.float32)
        zp[:self.Npml, :] = zp1
        zp[nz:, :] = zp1[::-1]

        a = self.dh * np.arange(self.Npml + 1, 1, -1).reshape(self.Npml, 1)
        xp1 = np.repeat(a, TNz).reshape(self.Npml, TNz).T
        xp = np.zeros((TNz, TNx), np.float32)
        xp[:, :self.Npml] = xp1
        a = xp1[:, ::-1]
        xp[:, nx:] = a

        if self.Npml != 0:

            d0 = (self.nd + 1) * np.log(1 / self.Rc) * v_max / (2 * self.Npml * self.dh)

            dz_pml = d0 * (zp / (self.Npml * self.dh)) ** self.nd
            dx_pml = d0 * (xp / (self.Npml * self.dh)) ** self.nd

            nuz = 1. + (self.nu0 - 1.) * (zp / (self.Npml * self.dh)) ** self.nnu
            nux = 1. + (self.nu0 - 1.) * (xp / (self.Npml * self.dh)) ** self.nnu

            alpha_z = self.alpha0 * (1. - (zp / (self.Npml * self.dh)) ** self.nalpha)
            alpha_x = self.alpha0 * (1. - (xp / (self.Npml * self.dh)) ** self.nalpha)

        else:

            dz_pml = np.zeros((TNz, TNx), np.float32)
            dx_pml = np.zeros((TNz, TNx), np.float32)

            nuz = 1. + np.zeros((TNz, TNx), np.float32)
            nux = 1. + np.zeros((TNz, TNx), np.float32)

            alpha_z = np.zeros((TNz, TNx), np.float32)
            alpha_x = np.zeros((TNz, TNx), np.float32)

        bz = np.exp(-(dz_pml * nuz + alpha_z) * self.dt)
        bx = np.exp(-(dx_pml * nux + alpha_x) * self.dt)

        with np.errstate(divide='ignore', invalid='ignore'):
            cz = dz_pml * nuz * (bz - 1.) / (dz_pml + alpha_z / nuz)
        cz[np.isnan(cz)] = 0.0

        with np.errstate(divide='ignore', invalid='ignore'):
            cx = dx_pml * nux * (bx - 1.) / (dx_pml + alpha_x / nux)
        cx[np.isnan(cx)] = 0.0

        self.bx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=bx)
        self.bz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=bz)
        self.cx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=cx)
        self.cz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=cz)
        self.nux_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                               self.mf.COPY_HOST_PTR, hostbuf=nux)
        self.nuz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                               self.mf.COPY_HOST_PTR, hostbuf=nuz)

        buufer_purpose = np.zeros((TNz, TNx), np.float32)

        self.psi_txxx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                    self.mf.COPY_HOST_PTR, hostbuf=buufer_purpose)
        self.psi_txzz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                    self.mf.COPY_HOST_PTR, hostbuf=buufer_purpose)
        self.psi_txzx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                    self.mf.COPY_HOST_PTR, hostbuf=buufer_purpose)
        self.psi_tzzz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                    self.mf.COPY_HOST_PTR, hostbuf=buufer_purpose)
        self.psi_vxx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=buufer_purpose)
        self.psi_vzz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=buufer_purpose)
        self.psi_vxz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=buufer_purpose)
        self.psi_vzx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=buufer_purpose)

        # pml for acoustic
        # dx_pml, dz_pml = pml_counstruction(TNz, TNx, self.dh, self.Npml, self.pmlR)
        vdx_pml = self.dx_pml * v_max
        vdz_pml = self.dz_pml * v_max

        self.vdx_pml_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=vdx_pml)
        self.vdz_pml_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=vdz_pml)

    def psi_reset(self, TNz, TNx):
        self.psi_txxx = np.zeros((TNz, TNx))
        self.psi_txzz = np.zeros((TNz, TNx))
        self.psi_txzx = np.zeros((TNz, TNx))
        self.psi_tzzz = np.zeros((TNz, TNx))
        self.psi_vxx = np.zeros((TNz, TNx))
        self.psi_vzz = np.zeros((TNz, TNx))
        self.psi_vxz = np.zeros((TNz, TNx))
        self.psi_vzx = np.zeros((TNz, TNx))


def pml_counstruction(TNz, TNx, dh,
                      n_pml=10, pml_r=1e-5, pml_dir=3):
    """
    pml_counstruction(TNz, TNx, dh, n_pml=10, pml_r=1e-5)

    PML construction generate two matrices for x- and z-directions with the
    size of velocity model plus number of pml samples in each direction.

    dx_pml and dz_pml are obtained based on Gao et al., 2017, Comparison of
    artiﬁcial absorbing boundaries for acoustic wave equation modelling.

    Parameters
    ----------
        TNz : int
            Number of samples in z-direction (`n_z + 2 * n_pml`).

        TNx : int
            Number of samples in x-direction (`n_x + 2 * n_pml`).

        dh : float
            Spatial ampling rate in x-direction.

        n_pml : int, optional = 10
            Number of pml layer

        pml_r : float, optional = 1e-5
            Theoretical reﬂection coefﬁcient.

    Returns
    --------
        dx_pml : float
            A matrix with the size of [TNz, TNx] with zero value
            everywhere excpete inside PML in right and left of model.

        dz_pml : float
                A matrix with the size of [TNz, TNx] with zero value
            everywhere excpet inside PML in above and bottom of model.

    References
    ----------
        [1] Gao et al., 2017, Comparison of artiﬁcial absorbing boundaries
        for acoustic wave equation modelling, Exploration Geophysics,
        2017, 48, 76–93.

        [2] Araujo and Pestana, 2020, Perfectly matched layer boundary conditions
        for the second-order acoustic wave equation solved by the rapid
        expansion method, Geophysical Prospecting, 2020, 68, 572–590.
    """
    dx_pml = np.zeros((TNz, TNx)).astype(np.float32, order='C')
    dz_pml = np.zeros((TNz, TNx)).astype(np.float32, order='C')

    # For x-direction
    a = pml_delta_calculation(dh, n_pml, pml_r)
    dx_pml[:, TNx - n_pml:] = a
    # np.fliplr(a.reshape(-1,1).T).reshape(-1)
    dx_pml[:, :n_pml] = np.flip(a, 0)

    # For z-direction
    a = pml_delta_calculation(dh, n_pml, pml_r)

    dz_pml[TNz - n_pml:, :] = a.reshape(-1, 1)
    dz_pml[:n_pml, :] = np.flip(a, 0).reshape(-1, 1)

    if pml_dir == 0:
        dx_pml = np.zeros(dx_pml.shape, np.float32)
    elif pml_dir == 1:
        dz_pml = np.zeros(dz_pml.shape, np.float32)
    elif pml_dir == 3:
        dz_pml[:n_pml, :] = np.zeros((len(a), dz_pml.shape[1]), dtype=np.float32)

    return dx_pml, dz_pml


def pml_delta_calculation(dh, n_pml=10, pml_r=1e-5):
    """
        pml_delta_calculation(n_pml, dh, pml_r)

        This function generates delta vector for PML construction function which put this vector
        around the model matrices.

        dx_pml and dz_pml are obtained based on Gao et al., 2017, Comparison of
        artiﬁcial absorbing boundaries for acoustic wave equation modelling.

        Warns
        -----
        TODO: I have to add dz as well

        Parameters
        ----------
            dh : float
                Sampling rate in x-direction.

            n_pml : int, optional = 10
                Number of pml layers

            pml_r : float, optional = 1e-5
                Theoretical reﬂection coefﬁcient.

        Returns
        --------
            delta : float
                A vector containing the absorbant value for putting in absorbant
                layer

        References
        ----------
            [1] Gao et al., 2017, Comparison of artiﬁcial absorbing boundaries
            for acoustic wave equation modelling, Exploration Geophysics,
            2017, 48, 76–93.

            [2] Araujo and Pestana, 2020, Perfectly matched layer boundary conditions
            for the second-order acoustic wave equation solved by the rapid
            expansion method, Geophysical Prospecting, 2020, 68, 572–590.
        """
    delta1 = n_pml * dh
    r = (np.arange(n_pml) * dh)
    if delta1 != 0:
        delta = np.float32(-(3 / (2 * delta1)) * ((r / delta1) ** 2) * (np.log10(pml_r)))
    else:
        delta = np.array([])
    return delta


def vel_dict2vec(m0):
    nz, nx = m0[[*m0][0]].shape
    m = np.zeros((3 * nz * nx))

    m[:nz * nx] = m0['vp'].reshape(-1)
    m[nz * nx: 2 * nz * nx] = m0['vs'].reshape(-1)
    m[2 * nz * nx:] = m0['rho'].reshape(-1)
    return m


def vec2vel_dict(m0, nz, nx):
    """
    vec2vel_dict converts a vector of DV to dictionary

    This function converts a vector of DV to dictionary which is
    used during the inversion.

    Args:
        m0 (1-d ndarray): a vector containg the whole parameters of the model
        nz ([type]): Number of samples of the model in z-direction
        nx ([type]): Number of samples of the model in x-direction

    Returns:
        m (dictionary): A dictionary ccontaining 'vp', 'vs', 'rho'.
    """
    m = {
        'vp': m0[:nz * nx].reshape(nz, nx),
        'vs': m0[nz * nx:2*nz * nx].reshape(nz, nx),
        'rho': m0[2*nz * nx:].reshape(nz, nx)
    }

    return m


def pcs_dict2vec(m0):
    """
    pcs_dict2vec converts a dictionary of PCS to a vector

    This function converts a dictionary of PCS to vector which is
    used during the inversion.

    Args:
        m0 (dictionary): A dictionary ccontaining 'phi', 'cc', 'sw'.

    Returns:
        m (dictionary): A vector containg the whole parameters of the model.
    """
    nz, nx = m0[[*m0][0]].shape
    m = np.zeros((3 * nz * nx))

    m[:nz * nx] = m0['phi'].reshape(-1)
    m[nz * nx: 2 * nz * nx] = m0['cc'].reshape(-1)
    m[2 * nz * nx:] = m0['sw'].reshape(-1)
    return m


def vec2pcs_dict(m0, nz, nx):
    """
    vec2pcs_dict converts a vector of PCS to dictionary

    This function converts a vector of PCS to dictionary which is
    used during the inversion.

    Args:
        m0 (1-d ndarray): a vector containg the whole parameters of the model
        nz ([type]): Number of samples of the model in z-direction
        nx ([type]): Number of samples of the model in x-direction

    Returns:
        m (dictionary): A dictionary ccontaining 'phi', 'cc', 'sw'.
    """
    m = {
        'phi': m0[:nz * nx].reshape(nz, nx),
        'cc': m0[nz * nx:2*nz * nx].reshape(nz, nx),
        'sw': m0[2*nz * nx:].reshape(nz, nx)
    }

    return m

def svd_reconstruction(m, begining_component, num_components):
    U, s, V = np.linalg.svd(m)
    reconst_img = np.matrix(U[:, begining_component:begining_component +num_components]) *\
        np.diag(s[begining_component:begining_component +num_components]) * \
            np.matrix(V[begining_component:begining_component +num_components, :])
    return reconst_img


def cost_preparation(dpre, dobs,
                     fn, freq=None, order=None, axis=None,
                     params_oh=None):
    """
    cost_preparation prepare the data for calculating the cost function

    This function prepare the data for calculating the cost function.
    This preparation is based on multi-scale inversion strategy (Bunks et al., 1995).

    Args:
        dpre ([type]): Predicted data
        dobs ([type]): Observed data
        fn (float): Nyquist frequency
        freq (float, optional): Desire frequency for filtering. Defaults to None.
        order (int, optional): Order of the filter. Defaults to None.
        axis (int, optional): Axis of the filter. Defaults to None.
        params_oh ([type], optional): Parameter to prepare the data for different offsets. Defaults to None.

    Returns:
        [type]: [description]
    """

    x_pre = copy.deepcopy(dpre)
    x_obs = copy.deepcopy(dobs)

    if freq:
        x_obs = lowpass(x_obs, freq, fn, order=order, axis=axis)
        x_pre = lowpass(x_pre, freq, fn, order=order, axis=axis)

    if params_oh is not None:
        x_obs = params_oh * x_obs
        x_pre = params_oh * x_pre

    return x_pre, x_obs


def lowpass(x1, highcut, fn, order=1, axis=1, show=False):
    x = copy.deepcopy(x1)

    # Zero padding
    padding = 512
    x = np.hstack((x, np.zeros((x.shape[0], padding, x.shape[2]))))

    nt = x.shape[axis]

    # Bring the data to frequency domain
    x_fft = fft.fft(x, n=nt, axis=axis)

    # Calculate the highcut btween 0 to 1
    scaled_highcut = 2*highcut/fn

    # Generate the filter
    b, a = butter(order, scaled_highcut, btype='lowpass', output="ba")

    # Get the frequency response
    w, h1 = freqz(b, a, worN=nt, whole=True)
    h = np.diag(h1)

    # Apply the filter in the frequency domain
    fd = h @ x_fft

    #Double filtering by the conjugate to make up the shift
    h = np.diag(np.conjugate(h1))
    fd = h @ fd

    # Bring back to time domaine
    f_inv = fft.ifft(fd, n=nt, axis=axis).real
    f_inv = f_inv[:, :-padding, :]

    return f_inv


def adj_lowpass(x, highcut, fn, order, axis=1):

    # Zero padding
    padding = 512
    x = np.hstack((x, np.zeros((x.shape[0], padding, x.shape[2]))))

    nt = x.shape[axis]

    # Bring the data to frequency domain
    x_fft = np.fft.fft(x, n=nt, axis=axis)

    # Calculate the highcut btween 0 to 1
    scaled_highcut = 2*highcut / fn

    # Generate the filter
    b, a = butter(order, scaled_highcut, btype='lowpass', output="ba")

    # Get the frequency response
    w, h = freqz(b, a, worN=nt, whole=True)

    # Get the conjugate of the filter
    h_c = np.diag(np.conjugate(h))

    # Apply the adjoint filter in the frequency domain
    fd = h_c @ x_fft

    # Double filtering by the conjugate to make up the shift
    h_c = np.diag(h)
    fd = h_c @ fd

    # Bring back to time domaine
    adj_f_inv = np.fft.ifft(fd, axis=axis).real
    adj_f_inv = adj_f_inv[:, :-padding, :]
    return adj_f_inv


def adj_cost_preparation(res,
                         fn, freq=False, order=None, axis=None,
                         params_oh=None):

    x_res = np.copy(res)

    if params_oh is not None:
        x_res = params_oh * x_res

    if freq:
        x_res = adj_lowpass(res, freq, fn, order=order, axis=axis)

    return x_res


def source_weighting(d_pre, d_obs, ns, nr):
    x_pre = np.copy(d_pre)

    alpha_res = np.zeros(x_pre.shape)
    for k in range(ns):
        alpha_res[:, k * nr: (k + 1) * nr] = \
            (x_pre[:, k * nr: (k + 1) * nr].reshape(-1).T @
             d_obs[:, k * nr: (k + 1) * nr].reshape(-1)) / \
            (x_pre[:, k * nr: (k + 1) * nr].reshape(-1).T @
             x_pre[:, k * nr: (k + 1) * nr].reshape(-1))

    return alpha_res


def cost_seismic(d_pre0, d_obs0, fun,
         fn=None, freq=None, order=None, axis=None,
         sourc_weight=False, ns=None, nr=None,
         params_oh=None):
    """
    cost_seismic calculates the cost between estimated and observed data.

    This function calculates the cost between estimated and observed data by applying desired filters
    and returns the cost and the adjoint of the residual.

    Args:
        d_pre0 (dict): Estimated data
        d_obs0 (dict): Observed data
        fun (function): The function to calculate the cost. This could be "CF = tools.CostFunction('l2')"
        fn (float, optional): Nyquist frequency. Defaults to None.
        freq (float, optional): Desired frequency to implement the lowpass filter. Defaults to None.
        order ([type], optional): [description]. Defaults to None.
        axis ([type], optional): [description]. Defaults to None.
        sourc_weight (bool, optional): [description]. Defaults to False.
        ns (int, optional): Number of the sources. Defaults to None.
        nr (int, optional): Number of the receivers. Defaults to None.
        params_oh ([type], optional): [description]. Defaults to None.

    Returns:
        rms (float): The cost
        adj_src: Adjoint source to propagate through the model in adjoint wave equation
    """


    d_pre = copy.deepcopy(d_pre0)
    d_obs = copy.deepcopy(d_obs0)

    dpre = np.array(list(d_pre.values()))
    dobs = np.array(list(d_obs.values()))
    # TODO: Treat as dict
    x_pre_cost, x_obs_cost = cost_preparation(dpre, dobs,
                                              fn, freq=freq, order=order, axis=axis,
                                              params_oh=params_oh)

    alpha_res = 1.0
    if sourc_weight:
        alpha_res = source_weighting(dpre, dobs, ns, nr)

    # for param in x_pre_cost:
    rms, res = fun(alpha_res*x_pre_cost, x_obs_cost)
    adj_src_ndarray = adj_cost_preparation(res, fn, freq=freq, order=order, axis=axis,
                                   params_oh=params_oh)

    adj_src = {}
    adj_src['vx'] = adj_src_ndarray[0, :, :].astype(np.float32)
    adj_src['vz'] = adj_src_ndarray[1, :, :].astype(np.float32)
    adj_src['taux'] = adj_src_ndarray[2, :, :].astype(np.float32)
    adj_src['tauz'] = adj_src_ndarray[3, :, :].astype(np.float32)
    adj_src['tauxz'] = adj_src_ndarray[4, :, :].astype(np.float32)

    return rms, adj_src


class CostFunction:
    """
    CostFunction provides different cost functions.


    Parameters
    ----------
    cost_function_type : str, optional
            Type of cost function, by default "l2"
    """
    def __init__(self, cost_function_type="l2"):
        self.cost_function_method = "self." + cost_function_type

    def __call__(self, dest, dobs):
        """
        By calling a CostFunction object, the loss is calculated.

        Parameters
        ----------
        dest : dict
            Estimated data
        dobs : dict
            Observed data

        Returns
        -------
        err : scalar float
            Error
        adj_src : dict
            A dictionary containing adjoint of the residuals
        """
        err, adj_src = eval(self.cost_function_method)(dest, dobs)
        return err, adj_src

    @staticmethod
    def list2dict(x):

        x_dict = {
            'vx': x[0, :, :],
            'vz': x[1, :, :],
            'taux': x[2, :, :],
            'tauz': x[3, :, :],
            'tauxz': x[4, :, :]
        }
        return x_dict

    def l1(self, dest, dobs):
        res = np.float32(dest - dobs)
        #TODO: adj_src is not right
        rms = np.sum(np.abs(res))
        adj_src = res  # np.ones(res.shape, np.float32)

        return rms, adj_src

    def l2(self, dest0, dobs0):

        dest = copy.deepcopy(dest0)
        dobs = copy.deepcopy(dobs0)

        if type(dest0).__name__ == 'ndarray':
            dest = self.list2dict(dest)
            dobs = self.list2dict(dobs)

        rms = 0
        res = {}
        for param in dest:
            res[param] = np.float32(dest[param] - dobs[param])
            rms += 0.5 * (res[param].reshape(-1).T @ res[param].reshape(-1))

        if type(dest0).__name__ == 'ndarray':
            adj_src = np.array(list(res.values()))
        else:
            adj_src = res

        return rms, adj_src

    def l2_intensity(self, dest, dobs):

        res = dest**2 - dobs**2
        rms = 0.25 * (res.reshape(-1).T @ res.reshape(-1))

        adj_src = dest * res
        return rms, adj_src

    def exponential_cost(self, dest, dobs):
        """
        based on
        https://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications

        """
        res = dest - dobs
        l2 = (res.reshape(-1).T @ res.reshape(-1))

        tau = 6.5  # 6.5  #TODO make it baed on the value of l2

        rms = tau * np.exp(l2/tau)
        adj_src = 2/tau * res * rms

        return rms, adj_src

    def l2_hilbert(self, dest, dobs):
        # Cost function based on envelope

        dobs_hilbert = hilbert(dobs, axis=-1)
        dest_hilbert = hilbert(dest, axis=-1)

        H_obs = np.imag(dobs_hilbert)
        H_est = np.imag(dest_hilbert)

        rms, adj_src_hilbert = self.l2(H_est, H_obs)

        # adjoint of real part is negative of itself
        adj_src = -1 * np.imag(hilbert(adj_src_hilbert, axis=-1))

        return rms, adj_src

    def l2_envelope(self, dest, dobs):
        """
        based on Wu et al., 2014, Seismic envelope inversion and modulation signal model
        Geiophysics
        """
        # Cost function based on envelope
        analytical_dobs = hilbert(dobs, axis=-1)
        analytical_dest = hilbert(dest, axis=-1)

        e_est = np.abs(analytical_dest)
        e_obs = np.abs(analytical_dobs)
        E = e_est - e_obs

        yest = np.real(analytical_dest)  # yest = dest
        yobs = np.real(analytical_dobs)  # yobs = dobs

        yH_est = np.imag(analytical_dest)
        yH_real = np.imag(analytical_dobs)

        rms = 0.5 * E.reshape(-1).T @  E.reshape(-1)
        adj_src = E * dest/e_est - np.imag(E*yH_est/e_est)
        # rms, adj_src_hilbert = self.l2(s_est, s_obs)
        #
        # adj_src = np.real(hilbert(adj_src_hilbert, axis=-1))

        """
        Plot to compare the envelope adjoint source with normal l2
        """
        SHOW = False
        if SHOW:
            trace_number = 30
            dt = 0.0006

            _, l2_src = self.l2(dest, dobs)
            self.plot_trace(l2_src[0, :, trace_number], adj_src[0, :, trace_number], "residual AS", "envelope AS")
            self.plot_trace(dobs[0, :, trace_number], e_obs[0, :, trace_number], "$d_{obs}$", "envelope")
            self.plot_amp_spectrum(l2_src[0, :, trace_number], adj_src[0, :, trace_number], dt,
                                   case_a_label="residual AS", case_b_label="envelope AS")

        return rms, adj_src

    def plot_trace(self, case_a, case_b, case_a_label=None, case_b_label=None):
        """
            to compare two trace

        """
        plt.figure()
        plt.plot(case_a, np.arange(case_a.size), label=case_a_label)
        plt.plot(case_b, np.arange(case_b.size), label=case_b_label)
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()

    def plot_amp_spectrum(self, case_a, case_b, dt, case_a_label=None, case_b_label=None):
        """
            to compare two amplitude spectrum

        """
        fdomain_a = np.abs(np.fft.fftshift(np.fft.fft(case_a)))

        fdomain_b = np.abs(np.fft.fftshift(np.fft.fft(case_b)))

        f_idx = np.linspace(-1/2/dt, 1/2/dt, fdomain_b.size)
        fig, ax = plt.subplots()
        ax.plot(f_idx, (fdomain_a/fdomain_a.max())**2, label=case_a_label)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized amplitude')
        plt.plot(f_idx, (fdomain_b/fdomain_b.max())**2, label=case_b_label)
        ax.legend()
        ax.set_xlim([0, 125])
        ax.grid()


def dict_diff(dict1, dict2, positivity=False):
    """
    dict_diff subtracts the contents of two dictionaries

    This function is used to subtract the parameters of a
    dictionary with the same parameter in another dictionary.

    Args:
        dict1 (dict): The first dictionary
        dict2 (dict): The second dictionary
        positivity (boolean, optional): A boolean variable to specify if the used wants
        to filter out the negative differences. Defaults to False.

    Returns:
        dic: A dictionary containing the common parameters of ```dict1``` and ```dict2```, but their difference.
    """
    diff_vel = {}
    for params in dict1:
        diff_vel[params] = (dict1[params] - dict2[params])
        if positivity:
            diff_vel[params][diff_vel[params] < 0] = 0
    return diff_vel


def dict_summation(dict1, dict2, division=1.0):
    """
    dict_summation add the contents of two dictionaries

    This function is used to add the parameters of a
    dictionary with the same parameter in another dictionary.

    Args:
        dict1 (dict): The first dictionary
        dict2 (dict): The second dictionary
        division (float, optional): In cas if user wants to devide the summation to a number (e.g. averaging)
        of amplifying the result. Defaults to 1.0.
    Returns:
        dic: A dictionary containing the common parameters of ```dict1``` and ```dict2```, but their summation.
    """
    sum_val = {}
    for params in dict1:
        sum_val[params] = (dict1[params] + dict2[params])/division

    return sum_val


def parameter_relation(m1, m2, order, idx=20, idx_test=-20, show=False):

    x = m1[:, idx]
    y = m2[:, idx]

    model = np.poly1d(np.polyfit(x, y, order))

    if show:

        x_test = m1[:, idx_test]
        y_test = m2[:, idx_test]

        y_test_pre = model(x_test)
        y_train_pre = model(x)

        fig = plt.figure(figsize=(8,4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, y, '*', label='True')
        ax.plot(x, y_train_pre, label='Predicted')
        res = y - y_train_pre
        l2 = np.linalg.norm(res, ord=2)
        ax.set_title(f'Training data rms: {l2:1.3}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.legend()
        ax.grid()

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x_test, y_test, '*', label='True')
        ax.plot(x_test, y_test_pre, 'o', label='Predicted')
        ax.plot([x_test, x_test], [y_test, y_test_pre], 'r--')
        res = y_test_pre - y_test
        l2 = np.linalg.norm(res, ord=2)
        ax.set_title(f'Test data rms: {l2:1.3}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.legend()
        ax.grid()
        plt.subplots_adjust(wspace=0.15)
        plt.show(block=False)

    return model


if __name__ == "__main__":
    R = Recorder(100, np.array([10]), 10, 1)
    print(R.vx.shape)
