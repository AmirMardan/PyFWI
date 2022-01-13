from PyFWI.wave_propagation import wave_propagator as Wave
from scipy.optimize.optimize import MemoizeJac
import PyFWI.optimization as opt
import PyFWI.fwi_tools as tools
import PyFWI.acquisition as acq
import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import axes_grid, make_axes_locatable
from scipy.optimize import line_search
import copy
import PyFWI.seiplot as splt

class FWI(Wave):
    """
    FWI implement full-waveform inversion (FWI)
    """
    def __init__(self, d_obs, inpa, src, rec_loc, model_size, n_well_rec, chpr, components, param_functions=None):
        """
        This class implement the FWI using the class 'wave_propagator'
        in 'PyFWI.wave_propagation'.

        Args:
            d_obs (dict): Observed data
            inpa (dict): Dictionary containg required parameters
            src (class): Source 
            rec_loc (ndarray): Location of the receivers
            model_size (ndarray): Size of the model
            n_well_rec ([type]): Number of receivers in the well in INPA['acq_type'] !=1
            chpr (int): Percentage of specify how much wavefield should be saved.
            components ([type]): Components of the operation
            param_functions ([type], optional): A list containg four function in case if the inversion is not happenning in DV parameterization. Defaults to None.
        """
        super().__init__(inpa, src, rec_loc, model_size, n_well_rec, chpr, components)

        if param_functions is None:
            self.dict2vec = tools.vel_dict2vec
            self.vec2dict = tools.vec2vel_dict
            self.to_dv = lambda a: a
            self.grad_from_dv = lambda a, b: a
        else:
            self.dict2vec = param_functions['dict2vec']
            self.vec2dict = param_functions['vec2dict']
            self.to_dv = param_functions['to_dv']
            self.grad_from_dv = param_functions['grad_from_dv']
            
        keys = inpa.keys()
        try:
            self.sd = inpa['sd']  # Virieux et al, 2009
        except:
            self.sd = 1.0
        self.d_obs = acq.prepare_residual(d_obs, 1)
        
        self.fn = inpa['fn']

        self.GN_wave_propagator = Wave(inpa, src, rec_loc, model_size, n_well_rec, chpr, components)

        if 'cost_function_type' in keys:
            self.CF = tools.CostFunction(inpa["cost_function_type"])
        else:
            self.CF = tools.CostFunction('l2')
        
        self.n_elements = self.nz * self.nx
        
    def __call__(self, m0, method, iter, freqs, n_params, k_0, k_end, video=False):
        """
        FWI implement the FWI

        Args:
            m0 (dict): The initial model
            method (int, str): The optimization method
            iter (ndarray): An array of iteration for each frequency
            freqs (float): Frequencies for multi-scale inversion.
            n_params (int): Number of parameter to invert for in each time
            k_0 (int): The first parameter of interest
            k_end (int): The last parameter of interest

        Returns:
            m_est (dict): The estimated model
            rms (ndarray): The rms error
        """
        m = self.dict2vec(m0)

        method = self.__fwi_method(method)
        m_video = np.empty(((k_end - k_0) * self.n_elements, len(freqs) + 1))
        
        m_video[:, 0] = m[(k_0 - 1) * self.n_elements: (k_end - 1) * self.n_elements]
        
        c = 0
        for freq in freqs:
            print(f"{freq = }")
            m, rms = eval(method)(m, iter[c], freq, n_params, k_0, k_end)
            
            c += 1
            m_video[:, c] = m[(k_0 - 1) * self.n_elements: (k_end - 1) * self.n_elements]
        
        if video:  
            return self.vec2dict(m, self.nz, self.nx), rms, m_video.reshape(self.nz, self.nx, len(freqs)+1)
        else:
            return self.vec2dict(m, self.nz, self.nx), rms

    def __fwi_method(self, user_method):
        
        method = 'self.'
        if user_method in [0, 'SD', 'sd']:
            raise ("Steepest descent is not provided yet.")
        elif user_method in [1, 'GD', 'gd']:
            raise ("Gradient descent is not provided yet.")
        elif user_method in [2, 'lbfgs']:
            method += 'lbfgs'
            
        return method
        
    def lbfgs(self, m0, ITER, freq, n_params=1, k0=0, k_end=1):
        # n_params: number of parameters to seek for in one iteration
                
        n_element = self.nz * self.nx
        mtotal = np.copy(m0)

        rms_hist = []

        fun = MemoizeJac(self.fprime_single)
        jac = fun.derivative
                
        for k in np.arange(k0-1, k_end-1, n_params):
            print(f'Parameter number {k + 1: } to {k + n_params: }')
                
            m_1 = mtotal[:k * n_element]
            m_opt = mtotal[k * n_element: (k + n_params) * n_element]
            m1 = mtotal[(k + n_params) * n_element:]  

            m_opt, hist, d = fmin_l_bfgs_b(fun, m_opt, jac, args=[m_1, m1, freq],
                                           m=10, factr=1e7, pgtol=1e-8, iprint=99,
                                           bounds=None, maxfun=15000, maxiter=ITER,
                                           disp=None, callback=None, maxls=20)

            # print(m_opt.max(), m_opt.min())
            rms_hist.append(hist)

            mtotal = np.hstack((m_1, m_opt, m1))
        return mtotal, rms_hist
    
    def fprime(self, m0, freq):
        
        mtotal = np.copy(m0)
        m_old = self.vec2dict(mtotal, self.nz, self.nx)
        m_new = self.to_dv(m_old)
        
        d_est = self.forward_modeling(m_new, show=False)
        d_est = acq.prepare_residual(d_est, self.sd)
        
        rms_data, adj_src = tools.cost_seismic(d_est, self.d_obs, fun=self.CF,
                                               fn=self.fn, freq=freq, order=3, axis=1
                                               )

        rms = rms_data

        grad_dv = self.gradient(adj_src, parameterization='dv')
        grad = self.grad_from_dv(grad_dv, m_old)
        
        grad = self.dict2vec(grad)
 
        return rms, grad
    
    def fprime_single(self, m0, m_1, m1, freq):
        mtotal = np.hstack((m_1, m0, m1))
        shape_1 = np.shape(m_1)[0]
        shape0 = np.shape(m0)[0]

        rms, grad = self.fprime(mtotal, freq)
        print(m0.min(), m0.max())
        print(" for f= {}: rms is: {}".format(freq, rms))

        return rms, grad[shape_1: shape_1 + shape0]
    

    