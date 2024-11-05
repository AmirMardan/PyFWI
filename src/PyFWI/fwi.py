from scipy.optimize.optimize import MemoizeJac
import numpy as np
from scipy.optimize import fmin_cg
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
from typing import List

from PyFWI.wave_propagation import WavePropagator as Wave
from PyFWI.fwi_tools import Regularization
import PyFWI.fwi_tools as tools
from PyFWI.processing import prepare_residual

class FWI(Wave):
    """
    FWI perform full-waveform inversion


    Parameters
    ----------
    d_obs : dict
        Observed data
    inpa : dict
        Input parameters
    src : Class
        Source object
    rec_loc : float
        Receiver location
    model_size : tuple
        Shape of the model
    n_well_rec : int
        Number of receivers in the well
    chpr : float (percentage)
        Percentage for check point 
    components : int
        Type of output
    param_functions : dict, optional
        List of functions required in case of inversion with different parameterization than dv, by default None
        """
    def __init__(self, d_obs, 
                 inpa, src, 
                 rec_loc, 
                 model_shape, 
                 components, 
                 chpr, 
                 n_well_rec=0, 
                 param_functions=None):
        super().__init__(inpa, src, rec_loc, model_shape, 
                         n_well_rec=n_well_rec, 
                         chpr=chpr, components=components)
        self.regularization = Regularization(self.nx, self.nz, self.dh, self.dh)
        
        if param_functions is None:
            self.dict2vec = tools.vel_dict2vec
            self.vec2dict = tools.vec2vel_dict
            self.model_to_dv = lambda a, param_functions_args: a
            self.grad_from_dv = lambda a, param_functions_args, b: a
            self.param_functions_args = []
        else:
            self.dict2vec = param_functions['dict2vec']
            self.vec2dict = param_functions['vec2dict']
            self.model_to_dv = param_functions['model_to_dv']
            self.grad_from_dv = param_functions['grad_from_dv']
            try:
                self.param_functions_args = param_functions['args']
            except:
                self.param_functions_args = []
            
        keys = inpa.keys()
        try:
            self.sd = inpa['sd']  # Virieux et al, 2009
        except:
            self.sd = 1.0
        
        try:
            self.param_relation = inpa['param_relation']
        except:
            self.param_relation = {}
        
        try:
            self.prior_model = inpa['prior_model']
        except:
            self.prior_model = None
            
            
        # Dictionnary for TV regularization
        if 'tv' in keys:
            self.tv_properties = inpa['tv']
        else:
            self.tv_properties = None
        
        if 'tikhonov' in keys:
            self.tikhonov_properties = inpa['tikhonov']
        else:
            self.tikhonov_properties = None
            
        self.d_obs = prepare_residual(d_obs, 1) 

        if 'cost_function_type' in keys:
            self.CF = tools.CostFunction(inpa["cost_function_type"])
        else:
            self.CF = tools.CostFunction('l2')
        
        if 'grad_coeff' in keys:
            self.grad_coeff = inpa['grad_coeff']
        else:
            self.grad_coeff = [1.0, 1.0, 1.0]
            
        self.n_elements = self.nz * self.nx
        
    def __call__(self, m0, method: str, 
                 iter: List[int], freqs: List[float], 
                 n_params, k_0, k_end):
        """
        Calling this object performs the FWI

        Parameters
        ----------
            m0 : dict
                The initial model
            method : str
                The optimization method. Either should be `cg` for congugate gradient or `lbfgs` for l-BFGS.
            iter : List[int]
                An array of iteration for each frequency
            freqs : List[float]
                Frequencies for multi-scale inversion.
            n_params : int
                Number of parameter to invert for in each time
            k_0 : int
                The first parameter of interest
            k_end : int
                The last parameter of interest

        Returns
        -------
            m_est : dictionary
                The estimated model
            rms : ndarray
                The rms error
        """
        m = self.dict2vec(m0)

        method = self.__fwi_method(method)

        c = 0
        rms = []
        
        k_0 = k_0 - 1
        k_end -= 1
        
        if k_0 + n_params > k_end:
            raise Exception("k_0 + n_params can't be larger than k_end!!")
        
        for freq in freqs:
            m, rms0 = eval(method)(m, iter[c], freq, n_params, k_0, k_end)
            
            rms.append(rms0)
            c += 1

        return self.vec2dict(m, self.nz, self.nx), np.array(rms)

    def __fwi_method(self, user_method):

        method = 'self.'
        if user_method in [0, 'SD', 'sd']:
            raise ("Steepest descent is not provided yet.")
            # TODO Add option
        elif user_method in [1, 'CG', 'cg']:
            method += 'cg'
            
        elif user_method in [2, 'lbfgs']:
            method += 'lbfgs'

        return method

    def cg(self, m0, ITER, freq, n_params=1, k0=1, k_end=2):
        
        n_element = self.nz * self.nx
        mtotal = np.copy(m0)

        rms_hist = []

        fun = MemoizeJac(self.fprime_single)
        jac = fun.derivative
        
        for k in np.arange(k0, k_end, n_params):
            print('Parameter number {} to {}'.format(k + 1, k + n_params))

            m_1 = mtotal[:k * n_element]
            m_opt = mtotal[k * n_element: (k + n_params) * n_element]
            m1 = mtotal[(k + n_params) * n_element:]
            
            m_opt, hist, _, _, _ = fmin_cg(fun, m_opt, jac, args=[m_1, m1, freq],
                                           gtol=1e-8, maxiter=ITER,
                                           full_output = True, disp=None)

            # print(m_opt.max(), m_opt.min())
            rms_hist.append(hist)

            mtotal = np.hstack((m_1, m_opt, m1))
        return mtotal, rms_hist
    
    def lbfgs(self, m0, ITER, freq, n_params=1, k0=1, k_end=2):
        
        n_element = self.nz * self.nx
        mtotal = np.copy(m0)

        rms_hist = []

        fun = MemoizeJac(self.fprime_single)
        jac = fun.derivative
        
        for k in np.arange(k0, k_end, n_params):
            print('Parameter number {} to {}'.format(k + 1, k + n_params))

            m_1 = mtotal[:k * n_element]
            m_opt = mtotal[k * n_element: (k + n_params) * n_element]
            m1 = mtotal[(k + n_params) * n_element:]

            m_opt, hist, d = fmin_l_bfgs_b(fun, m_opt, jac, args=[m_1, m1, freq],
                                           m=10, factr=1e7, pgtol=1e-8, iprint=99,
                                           bounds=None, maxfun=15000, maxiter=ITER,
                                           disp=None, callback=None, maxls=10)

            # print(m_opt.max(), m_opt.min())
            rms_hist.append(hist)

            mtotal = np.hstack((m_1, m_opt, m1))
        return mtotal, rms_hist

    def fprime(self, m0, freq):

        mtotal = np.copy(m0)
        m_old = self.vec2dict(mtotal, self.nz, self.nx)
        m_new = self.model_to_dv(m_old, self.param_functions_args)

        d_est = self.forward_modeling(m_new, show=False)
        d_est = prepare_residual(d_est, self.sd)

        rms_data, adj_src = tools.cost_seismic(d_est, self.d_obs, fun=self.CF,
                                               fn=self.fn, freq=freq, order=3, axis=1
                                               )
        
        grad_dv = self.gradient(adj_src, parameterization='dv')
        grad = self.grad_from_dv(grad_dv, self.param_functions_args, m_old)
        
        params = [*grad]
        grad[params[0]] *= self.grad_coeff[0]
        grad[params[1]] *= self.grad_coeff[1]
        grad[params[2]] *= self.grad_coeff[2]
        
        grad = self.dict2vec(grad)
        rms = rms_data 
        return rms, grad
    
    def fprime_single(self, m0, m_1, m1, freq):
        mtotal = np.float32(np.hstack((m_1, m0, m1)))
        shape_1 = np.shape(m_1)[0]
        shape0 = np.shape(m0)[0]
        
        k0 = np.int32(shape_1/self.n_elements)
        kend = np.int32(k0 + shape0/self.n_elements)
        
        rms_data, grad_data = self.fprime(mtotal, freq)
        
        rms_reg, grad_reg = self.regularization.cost_regularization(m0,
                                                  tv_properties=self.tv_properties,
                                                  tikhonov_properties=self.tikhonov_properties
                                                  )
        
        rms_model_relation, grad_model_relation = self.regularization.parameter_relation(mtotal, self.param_relation, k0, kend, freq)
        
        rms_mp, grad_mp = self.regularization.priori_regularization(m0, self.prior_model, k0, kend, freq)
        
        rms = rms_data + rms_reg + rms_model_relation + rms_mp
        grad = grad_data[shape_1: shape_1 + shape0] + \
            grad_reg + \
            grad_model_relation[shape_1: shape_1 + shape0] + \
            grad_mp
        
        print(m0.min(), m0.max())
        print(" for f= {}: rms is: {} with rms_reg: {}, and rms_data: {}, rms_mp: {}, rms_model_relation: {}".format(freq, rms, rms_reg, rms_data, rms_mp, rms_model_relation))

        return rms, grad
    