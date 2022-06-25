import copy
import logging
import numpy as np
from scipy.optimize.optimize import MemoizeJac
import matplotlib.pyplot as plt

from PyFWI.wave_propagation import WavePropagator as Wave
import PyFWI.fwi_tools as tools
import PyFWI.seiplot as splt
from PyFWI.processing import prepare_residual


def linesearch(fun, fprime, xk, pk, gk=None, fval_old=None, f_max=50, alpha0=None, show=False, min=1e-8, bond=[-np.inf, np.inf], args=()):

    x0 = copy.deepcopy(xk)
    rho = 0.5
    rho_inc = 0.5

    # Defining the initial alpha based on the ratio between data and gradient
    if alpha0 is None:
        alpha0 = np.abs(xk).max() / np.abs(pk).max()  # 1000.0

    initial_alpha = np.copy(alpha0)  # for comparing the alpha as a condition of increasing alpha
    fc = [0]
    max_call = f_max
    
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
    while (np.isnan(fval_new) or (fval_new >= fval_old)):  # & (count < max_call):
        alpha0 *= rho
        if alpha0 < min:
            alpha0 = 0
            break
        
        fval_new = phi(alpha0)
        # print(f"{alpha0 = } .......... {fval_new = :.6f} .......... {fval_old = :.6f}")
        count += 1

    if count == 0: # If we need to increase the alpha
        while (fval_new < fval_old) & (count < max_call):
            alpha_inc = alpha0 + rho_inc * alpha0
            fval_new_inc = phi(alpha_inc)

            count += 1
            # print(f"{alpha_inc = } .......... {fval_new_inc = :.6f} .......... {fval_old = :.6f}")
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
    
    fval_new = phi(alpha)
    grad_new = dephi(alpha) 
    
    # print(f'{fval_new = } -------------------------------{fval_old = } with {alpha = } with  {count = }')
    
    return alpha, fval_new, grad_new


class FWI(Wave):
    def __init__(self, d_obs, inpa, src, rec_loc, model_size, n_well_rec, chpr, components):
        super().__init__(inpa, src, rec_loc, model_size, n_well_rec, chpr, components)

        keys = inpa.keys()
        
        self.d_obs = d_obs
        self.fn = inpa['fn']
        try:
            self.sd = inpa['sd']  # Virieux et al, 2009
        except:
            self.sd = 1.0
        self.GN_wave_propagator = Wave(inpa, src, rec_loc, model_size, n_well_rec, chpr, components)
        
        if 'cost_function_type' in keys:
            self.CF = tools.CostFunction(inpa["cost_function_type"])
        else:
            self.CF = tools.CostFunction('l2')
        
    def __call__(self, m0, method, iter, freqs):
        m = tools.vel_dict2vec(m0)
        
        if method in [0, 'sd', 'SD']:
            m1, rms = self.steepest_descent(m, iter, freqs)
        if method in [3, 'gn', 'GN']:
            m1, rms = self.gauss_newton(m, iter, freqs)

        return tools.vec2vel_dict(m1, self.nz, self.nx), rms
    
    def run(self, m0, method, iter, freqs, n_params=3, k_0=1, k_end=4):
        m, rms = self(m0, method, iter, freqs)
        return m, rms
    
    def steepest_descent(self, m0, iter, freqs):
        
        m_opt = m0#[:self.nz*self.nx]
        m1 = np.array([]) # m0[self.nz*self.nx:] #
        
        rms_hist = []
        
        fun = MemoizeJac(self.fprime)
        jac = fun.derivative
        
        alpha = [10., 10., 10.]
        for freq in freqs:
            for i in range(iter):
                # print(f"Iteration === {i:1d}")
                rms, grad = self.fprime(m_opt, m1, freq)
                rms_hist.append(rms)
                
                p = -1.0 * grad
                
                mtotal, alpha = self.parameter_optimization(m_opt, m1, p, rms, grad, alpha, freq)
        
        return mtotal, rms_hist
    
    def gauss_newton(self, m0, iter, freqs):
        # GN = GaussNewton(self.GN_wave_propagator)
        rms_hist = []
        
        m_opt = m0
        m1 = np.array([])
        
        fun = MemoizeJac(self.fprime)
        jac = fun.derivative
        
        rms, grad = self.fprime(m_opt, m1, freqs[0])
        #TODO loop over freqs
        dp0 = np.array([0])
        alpha = [10., 10., 10.]
        
        i = 0
        while i < iter[0]:
            print("Iteration === {}".format(i))
            i += 1
            
            rms_hist.append(rms)
            
            p = truncated(self.GN_wave_propagator, self.W, m_opt, grad, m1, iter=5)
            splt.gn_plot(p, grad, self.nz, self.nx)

            mtotal, alpha = self.parameter_optimization(m_opt, m1, p, rms, grad, alpha, freqs[0])
        return mtotal, rms_hist
    
    def fprime(self, m0, m1, freq):
        mtotal = np.hstack((m0, m1))
        
        m = tools.vec2vel_dict(mtotal, self.nz, self.nx)
        
        d_est = self.forward_modeling(m, show=False)
        d_est = prepare_residual(d_est, self.sd)
        rms_data, adj_src = tools.cost_seismic(d_est, self.d_obs, fun=self.CF,
                                               fn=self.fn, freq=freq, order=3, axis=1
                                               )
        rms_data, adj_src = self.CF(d_est, self.d_obs)
        rms = rms_data
        
        g = self.gradient(adj_src)
        
        grad = tools.vel_dict2vec(g)  # [:self.nz * self.nx]
        
        return rms, grad
    
    
    def fprime_single(self, m0, m_1, m1, freq):
        mtotal = np.hstack((m_1, m0, m1))
        
        rms, grad = self.fprime(mtotal, np.array([]), freq)
        
        return rms, grad

    def parameter_optimization(self, m_opt, m1, p, rms, grad, alpha, freq):
        for j in range(3):
            fun_single = MemoizeJac(self.fprime_single)
            jac_single = fun_single.derivative

            n_element = self.nz*self.nx
            
            m_opt_1 = m_opt[:(j)*n_element]
            m_opt1 = m_opt[j*n_element:(j+1)*n_element]
            m11 = m_opt[(j+1)*n_element:] 
                
            p1 = p[j*n_element:(j+1)*n_element]
            grad1 = grad[j*n_element:(j+1)*n_element]
                
            alpha[j], rms, grad = linesearch(fun_single, jac_single, m_opt1, p1, grad1, rms, f_max=50,
                                             alpha0=alpha[j], args=[m_opt_1, m11, freq])
                
            m_opt[j*n_element:(j+1)*n_element] += alpha[j] * p[j*n_element:(j+1)*n_element]  # p_switched
        
        mtotal = np.hstack((m_opt, m1))
        
        return mtotal, alpha
    

def truncated(FO_waves, W, m0, grad0, m1, iter):
        nz = FO_waves.nz
        nx = FO_waves.nx
        n_params = np.int32(grad0.shape[0]/(nz * nx))
        
        a1 = 0.5
        a2 = 1.5 
        
        etta = 0.9
        Q = np.eye(n_params *nz * nx, n_params * nz * nx)
        
        m = tools.vec2vel_dict(np.hstack((m0, m1)), nz, nx)
        
        r = np.copy(grad0)
        y = np.dot(Q, grad0)
        x = -1 * np.copy(r)
                
        x_dict = tools.vec2vel_dict(x, nz, nx)
        
        Hx = 0
        
        dp = 0
        
        i = 0
        
        df_k_1 = np.linalg.norm(r, 2)
        
        while np.linalg.norm(Hx + r, 2) > etta * np.linalg.norm(r, 2) and (i <iter):
            data_section_ajoint = FO_waves.forward_modeling(m, False, W, x_dict)
            FO_waves.W = copy.deepcopy(W)
            hess = FO_waves.gradient(data_section_ajoint, Lam=None, grad=None, show=True)
            Hx = tools.vel_dict2vec(hess)
            
            b1 = np.dot(Hx.T, x)
            # print(f'{b1 = }')
            if b1<0:
                if np.all(dp == 0):
                    dp = x
                break
            b2 = np.dot(r, y)
            
            b2b1 = (b1/b2)  # The fraction is reverse in Metivier
            
            dp += b2b1 * x  
            
            r = r + b2b1 * Hx
            y = np.dot(Q, r)
            
            x = -r + (np.dot(r, y)/b2) * x
            x_dict = tools.vec2vel_dict(x, nz, nx)
           
            i += 1
            
            df_k = np.linalg.norm(r, 2)
            
            etta = a1 * (df_k / df_k_1) ** a2
            
            df_k_1 = np.copy(df_k)
            
        return dp
        
        