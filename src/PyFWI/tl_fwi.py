from PyFWI.wave_propagation import wave_propagator as Wave
from scipy.optimize.optimize import MemoizeJac
import PyFWI.fwi_tools as tools
import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from scipy.optimize import line_search
import copy
from PyFWI.fwi import FWI
import PyFWI.acquisition as acq
import timeit


class TimeLapse(Wave):
    def __init__(self, b_dobs, m_dobs, inpa, src, rec_loc, model_size, n_well_rec, chpr, components, param_functions=None):
        super().__init__(inpa, src, rec_loc, model_size, n_well_rec, chpr, components)
        
        self.inv_obj = FWI(b_dobs, inpa, src, rec_loc, model_size, n_well_rec, chpr, components, param_functions)

        self.b_dobs = copy.deepcopy(b_dobs)
        self.m_dobs = copy.deepcopy(m_dobs)
        
        # self.nz, self.nx = model_size
        self.n_elements = self.nz * self.nx
        
        self.CF = self.inv_obj.CF
        self.fn = self.inv_obj.fn
            
    def __call__ (self, b_m0, iter, freqs, tl_method, n_params, k_0, k_end):
        b_m0 = self.inv_obj.dict2vec(b_m0)
                
        if tl_method == 'cc':
            m, rms = self.cascaded_inversion(b_m0, freqs, iter, n_params=n_params, k_0=k_0, k_end=k_end)
        elif tl_method == 'sim':
            m, rms = self.sim_inversion(b_m0, freqs, iter, data_to_invert=self.b_dobs, n_params=n_params, k_0=k_0, k_end=k_end)
        elif tl_method == 'wa':
            m, rms = self.weighted_averaged_cascaded(b_m0, freqs, iter, n_params=n_params, k_0=k_0, k_end=k_end)
        elif tl_method == 'cj':
            m, rms = self.cascaded_joint(b_m0, freqs, iter, n_params=n_params, k_0=k_0, k_end=k_end)
        elif tl_method == 'cd':
            m, rms = self.central_difference(b_m0, freqs, iter, n_params=n_params, k_0=k_0, k_end=k_end)
        elif tl_method == 'cu':
            m, rms = self.cross_updating(b_m0, freqs, iter, n_params=n_params, k_0=k_0, k_end=k_end)
        return m, rms
    
    def tl_lbfgs(self, x0, iter, freq, n_params, k0, k_end):
        n_elements = self.nz * self.nx
        xtotal = np.copy(x0)
        self.xrest = np.copy(x0)
        
        rms_hist = []

        fun = MemoizeJac(self.tl_fprime)
        jac = fun.derivative
        
        c = 0
        for k in np.arange(k0-1, k_end-1, n_params):
            print('Parameter number {} to {}'.format(k + 1, k + n_params))
                            
            m_opt = np.hstack((
                x0[k * n_elements: (k + n_params) * n_elements],
                x0[(3 + k) * n_elements: 3*n_elements + (k + n_params) * n_elements],
            ))
            
            m_opt, hist, d = fmin_l_bfgs_b(fun, m_opt, jac, args=[freq, k, n_params], m=30,
                                           factr=1e-10, pgtol=1e-12, iprint=99, bounds=None,
                                           maxfun=15000, maxiter=iter, disp=None,
                                           callback=None, maxls=20)

            print(m_opt.max(), m_opt.min())
            rms_hist.append(hist)

            x0[k * n_elements:(k + n_params) * n_elements] = \
                np.copy(np.float32(m_opt[:n_params*n_elements]))

            x0[(3 + k) * n_elements:(3 + k + n_params)*n_elements] = \
                np.copy(np.float32(m_opt[n_params*n_elements:]))

        return x0, rms_hist 
    
    def tl_fprime(self, x, freq, k, n_params):
        """
        This function calculate the gradient of cost function

        Parameters:
        -----------
        x: float32
            This matrix with the shape of [number of parameters, nz*nx] contains
            the parameters
        """
        x1 = self.xrest

        x1[k * self.n_elements:(k + n_params) * self.n_elements] = x[:n_params * self.n_elements]

        x1[3*self.n_elements + k * self.n_elements: 3*self.n_elements + (k + n_params) * self.n_elements] = \
            x[n_params * self.n_elements:2*n_params*self.n_elements]

        b_model0_pcs = self.inv_obj.vec2dict(x1[:3*self.n_elements], self.nz, self.nx)
        m_model0_pcs = self.inv_obj.vec2dict(x1[3*self.n_elements:], self.nz, self.nx)

        b_model0 = self.inv_obj.to_dv(b_model0_pcs, [])
        m_model0 = self.inv_obj.to_dv(m_model0_pcs, [])
        
        dpre = self.forward_modeling(b_model0, show=False)
        dpre = acq.prepare_residual(dpre, self.inv_obj.sd)
        
        b_rms_data, adj_src1 = tools.cost_seismic(d_pre0=dpre, d_obs0=self.b_dobs, fun=self.CF,
                                         fn=self.fn, freq=freq, order=3, axis=1,
                                         sourc_weight=None, ns=self.ns, nr=self.nr,
                                         # params_oh=self.offset_weight[params_oh]
                                         )
        
        b = x1[k * self.n_elements:(k + n_params) * self.n_elements]
        m = x1[(3 + k)*self.n_elements:(3 + k + n_params) * self.n_elements]
        
        diff = m - b
        
        rms_diff = 0.5 * np.dot(diff.T, diff)
        
        gamma = 1e-10
        grad_diff_b = b - m
        grad_diff_m = m - b
        
        b_grad_dv = self.gradient(adj_src1)
        b_grad = self.inv_obj.grad_from_dv(b_grad_dv, [], b_model0_pcs)
        
        dpre = self.forward_modeling(m_model0, show=False)
        dpre = acq.prepare_residual(dpre, self.inv_obj.sd)
        
        m_rms_data, adj_src = tools.cost_seismic(d_pre0=dpre, d_obs0=self.m_dobs, fun=self.CF,
                                           fn=self.fn, freq=freq, order=3, axis=1,
                                           sourc_weight=None, ns=self.ns, nr=self.nr,
                                           # params_oh=self.offset_weight[params_oh]
                                           )

        m_grad_dv = self.gradient(adj_src)
        m_grad = self.inv_obj.grad_from_dv(m_grad_dv, [],m_model0_pcs)
        
        rms_data = b_rms_data + m_rms_data
        
        rms = rms_data + gamma * rms_diff 
        
        b_grad = self.inv_obj.dict2vec(b_grad) 
        m_grad = self.inv_obj.dict2vec(m_grad)
        
        # diff = np.abs(x[:self.n_elements] - x[self.n_elements:])
        print(k, np.abs(diff).min(), np.abs(diff).max())
        print("TL fprime_single_parameter for f= {}: rms is: {} for data: {} and model: {}".format(freq, rms, rms_data, rms_diff))
        
        grad = np.hstack((1 * b_grad[k * self.n_elements:(k + n_params) * self.n_elements] + gamma * grad_diff_b,
                          1 * m_grad[k * self.n_elements:(k + n_params) * self.n_elements] + gamma * grad_diff_m,
                          ))

        return rms, np.float64(grad)

    def weighted_averaged_cascaded(self, b_m0, freqs, iter, n_params, k_0, k_end):
        
        m0 = copy.deepcopy(b_m0)
        
        print("*** First inversion ***")
        self.inv_obj.d_obs = self.b_dobs
        model_b1, rms1 = self.inv_obj(m0, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)

        print("*** Second inversion ***")
        self.inv_obj.d_obs = self.m_dobs
        model_m1, rms2 = self.inv_obj(model_b1, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)

        print("*** Third inversion ***")
        self.inv_obj.d_obs = self.b_dobs
        model_b2, rms3 = self.inv_obj(model_m1, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)
        
        inverted = {
            "dm11": tools.dict_diff(model_m1, model_b1),
            "dm12": tools.dict_diff(model_m1, model_b2),
            "m1": model_m1,
            "b1": model_b1,
            "b2": model_b2,
        }
        print("improved_cascaded")
        return inverted, np.array(rms1 + rms2 + rms3) / 3
    
    def cross_updating(self, b_m0, freqs, iter, n_params, k_0, k_end):
        """
            Maharramov and Biondi, 2014
        """
        m0 = copy.deepcopy(b_m0)
        print("*** First inversion ***")
        tic = timeit.default_timer()
        self.inv_obj.d_obs = self.b_dobs
        model_b1, rms1 = self.inv_obj(m0, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)

        print("*** Second inversion ***")
        self.inv_obj.d_obs = self.m_dobs
        model_m1, rms2 = self.inv_obj(model_b1, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)
        process_time_cc = timeit.default_timer() - tic
        
        print("*** Third inversion ***")
        self.inv_obj.d_obs = self.b_dobs
        model_b2, rms3 = self.inv_obj(model_m1, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)
        process_time_wa = timeit.default_timer() - tic
        
        print("*** Fourth inversion ***")
        self.inv_obj.d_obs = self.m_dobs
        model_m2, rms4 = self.inv_obj(model_b2, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)
        process_time_cu = timeit.default_timer() - tic

        inverted = {
            "dm11": tools.dict_diff(model_m1, model_b1),
            "dm12": tools.dict_diff(model_m1, model_b2),
            "dm21": tools.dict_diff(model_m2, model_b1),
            "dm22": tools.dict_diff(model_m2, model_b2),
            "b1": model_b1,
            "m1": model_m1,
            "b2": model_b2,
            "m2": model_m2,
            "pt_cc": process_time_cc,
            "pt_wa": process_time_wa,
            "pt_cu": process_time_cu
        }
        print("cross_updating")
        return inverted, np.array(rms1 + rms2 + rms3 + rms4) / 4
    
    
    def central_difference(self, b_m0, freqs, iter, n_params, k_0, k_end):
        
        m0 = copy.deepcopy(b_m0)
            # ============ Central difference for parallel ==============

        print("*** Central time lapse *** ")
        self.inv_obj.d_obs = self.b_dobs
        model_b1, rms1 = self.inv_obj(m0, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)

        print("*** Second INVERSION ***")
        self.inv_obj.d_obs = self.m_dobs
        model_b1m, _ = self.inv_obj(model_b1, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)

        print("*** Third INVERSION ***")
        self.inv_obj.d_obs = self.m_dobs
        model_m1, _ = self.inv_obj(m0, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)

        print("*** Fourth INVERSION ***")
        self.inv_obj.d_obs = self.b_dobs
        model_m1b, _ = self.inv_obj(model_m1, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)

        inverted = {
            "dm0": tools.dict_diff(model_m1,  model_b1),
            "dm1": tools.dict_diff(model_b1m,  model_m1b),
            "dm3": tools.dict_diff(model_b1m, model_b1),
            "dm4": tools.dict_diff(model_m1, model_m1b),
            "dm34": tools.dict_summation(tools.dict_diff(model_b1m, model_b1),
                                 tools.dict_diff(model_m1, model_m1b),
                                division=2)
        }
        print("central_inversion")
        return inverted, _
        
    def cascaded_joint(self, m0, freqs, iter, n_params, k_0, k_end):
        """
            
        """
        # ============ Central difference for sim ==============
        print("*** FIRST INVERSION ***")
        model1, rms1 = self.sim_inversion(m0, freqs, iter=iter,n_params=n_params,
                                       k_0=k_0, k_end=k_end, data_to_invert=None)
        model_b1 = model1["b1"]
        model_m1 = model1["m1"]

        ave_model = tools.dict_summation(model_b1, model_m1, division=2)

        print("*** SECOND INVERSION ***")
        model_b1x, rms2 = self.sim_inversion(ave_model, freqs, iter=iter,n_params=n_params,
                                       k_0=k_0, k_end=k_end, data_to_invert=None)
        model_b1b = model_b1x["b1"]
        model_b1m = model_b1x["m1"]

        inverted = {
            "dm0": tools.dict_diff(model_m1,  model_b1),
            "dm7": tools.dict_diff(model_b1m, model_b1b),

            "dm3": tools.dict_diff(model_b1m, model_b1),
            "dm6": tools.dict_diff(model_m1, model_b1b),

            "model_b1": model_b1,
            "model_m1": model_m1,

            "model_b2": model_b1b,
            "model_m2": model_b1m,
        }

        print("joint_cascaded")
        return inverted, (rms1 + rms2) / 2
    
    def cascaded_inversion(self, b_m0, freqs, iter, n_params, k_0, k_end):

        self.inv_obj.d_obs = self.b_dobs
        model_b1, rms1 = self.inv_obj(b_m0, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)
        initial = model_b1.copy()

        print('===== Inversion of baseline is done ======')
        
        self.inv_obj.d_obs = self.m_dobs
        model_m1, rms2 = self.inv_obj(model_b1, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)

        inverted = {
            "b1": model_b1,
            "m1": model_m1,
            "dm":  tools.dict_diff(model_m1, initial, positivity=False)
        }
        print("Cascaded")

        hist = rms1 + rms2

        return inverted, np.array(hist)
    
    def sim_inversion(self, b_m0, freqs, iter, n_params, k_0, k_end, data_to_invert=None):

        hist = []  

        if data_to_invert is not None:
            self.dobs = data_to_invert
            model_ref0, _ = self.inv_obj(b_m0, method=2, freqs=freqs, iter=iter, n_params=n_params, k_0=k_0, k_end=k_end)

            b_model0 = copy.deepcopy(model_ref0)
            m_model0 = copy.deepcopy(model_ref0)

        else:
            b_model0 = copy.deepcopy(b_m0)
            m_model0 = copy.deepcopy(b_m0)
            

        x0 = np.hstack((self.inv_obj.dict2vec(b_model0),
                        self.inv_obj.dict2vec(m_model0),
                        ))

        # method = self.optimization_method(operation="time-lapse")

        iter_num = 0
        c = 0
        for freq in freqs:
            self.restForTest = x0

            x0, hist1 = self.tl_lbfgs(x0, iter[c], freq, n_params, k_0, k_end)

            hist.append(hist1[0])
            c += 1
        
        models = self.ins_TL_output(x0, False)
        inverted = {
            "dm": self.ins_TL_output(x0, True),
            "b1": models["model_b"],
            "m1": models["model_m"]
        }

        return inverted, np.array(hist/max(hist))
    
    
    def ins_TL_output(self, xk, diff=True):
        """
        This function make data which exist in the form of vectorial, in dictionary form.

        Parameters:
        -----------
            xk: flaot32
                the data as the size of [number_of_elements, nz*nx]

            nz: integer
                Number of samples in z-direction

            nx: integer
                Number of samples in x-direction

        Returns:
        --------
            model: dictonary
                A dictionary containg the model parameters

        """
        tn_elements = int(xk.shape[0]/2)
        if diff:
            x = xk[tn_elements: ] - xk[:tn_elements]
            model = self.inv_obj.vec2dict(x, self.nz, self.nx)
        else:
            model = {
                "model_b": self.inv_obj.vec2dict(xk[:tn_elements], self.nz, self.nx),
                "model_m": self.inv_obj.vec2dict(xk[tn_elements: 2*tn_elements], self.nz, self.nx)
            }
        return model   
    