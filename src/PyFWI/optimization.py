import copy
import logging
import numpy as np
from PyFWI.wave_propagation import wave_propagator as Wave, truncated
import PyFWI.fwi_tools as tools
from scipy.optimize.optimize import MemoizeJac


def linesearch(fun, fprime, xk, pk, gk=None, fval_old=None, f_max=50, alpha0=None, args=()):

    x0 = copy.deepcopy(xk)
    rho = 0.5
    rho_inc = 0.5

    # Defining the initial alpha based on the ratio between data and gradient
    if alpha0 is None:
        alpha0 = np.abs(xk).max() / np.abs(pk).max()  # 1000.0

    initial_alpha = np.copy(alpha0)  # for comparing the alpha as a condition of increasing alpha
    fc = [0]
    max_call = 15
    
    def phi(alpha):
        fc[0] += 1
        x1 = x0 + alpha * pk
        return fun(x1, *args)
    
    def dephi(alpha):
        x1 = x0 + alpha * pk
        return fprime(x1)
    
    if fval_old is None:
        fval_old = phi(0.0)
    
    fval_new = phi(alpha0)
    
    count = 0

    # For decreasing the alpha
    while (np.isnan(fval_new) or (fval_new > fval_old)):  # & (count < max_call):
        alpha0 *= rho
        fval_new = phi(alpha0)
        
        print(f"{alpha0 = } .......... {fval_new = :.4f} .......... {fval_old = :.4f}")
        count += 1

    if count == 0: # If we need to increase the alpha
        while (fval_new < fval_old) & (count < max_call):
            alpha_inc = alpha0 + rho_inc * alpha0
            fval_new_inc = phi(alpha_inc)

            count += 1
            print(f"{alpha_inc = } .......... {fval_new_inc = :.4f} .......... {fval_old = :.4f}")
            if fval_new_inc < fval_new:
                alpha0 = np.copy(alpha_inc)
                fval_new = np.copy(fval_new_inc)
            else:
                break

    if (fval_new > fval_old) & (count == max_call):
        alpha = 0
        logging.warning("Linesearch didn't converge.")
    else:
        alpha = alpha0 
    print(f'{initial_alpha = } -------------------------------{alpha = :.4f} with  {count = }')
    
    return alpha, phi(alpha), dephi(alpha)


class FWI(Wave):
    def __init__(self, d_obs, inpa, src, rec_loc, model_size, n_well_rec, chpr, components):
        super().__init__(inpa, src, rec_loc, model_size, n_well_rec, chpr, components)

        self.d_obs = d_obs
        
        self.GN_wave_propagator = Wave(inpa, src, rec_loc, model_size, n_well_rec, chpr, components)
        
    def __call__(self, m0, method, iter):
        m = tools.vel_dict2vec(m0)
        
        if method in [0, 'sd', 'SD']:
            m1, rms = self.steepest_descent(m, iter)
        if method in [1, 'gn', 'GN']:
            m1, rms = self.gauss_newton(m, iter)

        return tools.vec2vel_dict(m1, self.nz, self.nx), rms
    
    def steepest_descent(self, m0, iter):
        
        m_opt = m0[:self.nz*self.nx]
        m1 = m0[self.nz*self.nx:] # np.array([]) #
        
        rms_hist  = []
        
        fun = MemoizeJac(self.fprime)
        jac = fun.derivative
        
        alpha = None
        for i in range(iter):
            rms, grad = self.fprime(m_opt, m1)
            rms_hist.append(rms)
            
            p = -1.0 * grad
            alpha, _, _ = linesearch(fun, jac, m_opt, p, grad, rms, f_max= 50, alpha0=alpha, args=[m1])
            
            m_opt += alpha * p
        
        mtotal = np.hstack((m_opt, m1))
        return mtotal, rms_hist
    
    def gauss_newton(self, m0, iter):
        # GN = GaussNewton(self.GN_wave_propagator)
        rms_hist  = []
        
        m_opt = m0#[:self.nz*self.nx]
        m1 =  np.array([]) # m0[self.nz*self.nx:] #
        
        fun = MemoizeJac(self.fprime)
        jac = fun.derivative
        
        alpha = 16.
        rms, grad = self.fprime(m_opt, m1)
        # rms_hist.append(rms)
        
        dp0 = np.array([0])
        alpha = [.0001, .0001, .0001]
        
        i = 0
        while (i<iter):# and (rms>0.001):
            i += 1
            
            rms_hist.append(rms)
            
            # p = -1.0 * grad
            p = truncated(self.GN_wave_propagator, self.W, m_opt, grad, m1)
            
            for j in range(3):
                fun_single = MemoizeJac(self.fprime_single)
                jac_single = fun_single.derivative

                n_element = self.nz*self.nx
                m_opt_1 = m_opt[:(j)*n_element]
                m_opt1 = m_opt[j*n_element:(j+1)*n_element]
                m11 =   m_opt[(j+1)*n_element:]  # np.array([]) #  # 
                p1 = p[j*n_element:(j+1)*n_element]
                grad1 = grad[j*n_element:(j+1)*n_element]
                
                alpha[j], rms, grad = linesearch(fun_single, jac_single, m_opt1, p1, grad1, rms, f_max= 50, alpha0=alpha[j], args=[m_opt_1, m11])
                
                m_opt[j*n_element:(j+1)*n_element] += alpha[j] * p[j*n_element:(j+1)*n_element]  # p_switched
        
            mtotal = np.hstack((m_opt, m1))
        
        return mtotal, rms_hist
    
    def fprime(self, m0, m1):
        mtotal = np.hstack((m0, m1))
        
        m = tools.vec2vel_dict(mtotal, self.nz, self.nx)
        
        d_est = self.forward_modeling(m, show=False)
        
        res = tools.residual(d_est, self.d_obs)
        
        rms = tools.cost_function(d_est, self.d_obs)
        
        g = self.gradient(res)
        
        grad = tools.vel_dict2vec(g)  #[:self.nz * self.nx]
        
        return rms, grad
    
    
    def fprime_single(self, m0, m_1, m1):
        mtotal = np.hstack((m_1, m0, m1))
        
        m = tools.vec2vel_dict(mtotal, self.nz, self.nx)
        
        d_est = self.forward_modeling(m, show=False)
        
        res = tools.residual(d_est, self.d_obs)
        
        rms = tools.cost_function(d_est, self.d_obs)
        
        g = self.gradient(res)
        
        grad = tools.vel_dict2vec(g)#[:self.nz * self.nx]
        
        return rms, grad
