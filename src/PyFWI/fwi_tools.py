import os 
import numpy as np
import copy 
import logging
import pyopencl as cl
from numpy.core.arrayprint import dtype_is_implied 
from scipy.signal import butter, hilbert, freqz
import numpy.fft as fft
import matplotlib.pyplot as plt

try:
    from PyFWI.seismic_io import load_mat
    from PyFWI import rock_physics as rp 
except:
    from seismic_io import load_mat
    import rock_physics as rp 
import PyFWI.processing as seis_process


def inpa_generator(vp, sdo, fn, **kwargs):
    D = seis_process.derivatives(order=sdo)
    dh = vp.min()/(D.dh_n * fn)
    
    dt = D.dt_computation(vp.max(), dh)
    
    
    inpa = {
        "SeisCL": False,
        "seisout": 4,
        "no_use_GPUs": np.array([-1]),
        "cost_function_type": "l2",
        "cost_function_intensity": "l2_intensity",
        "device": 0,
        "medium": 1,
        "ns": 1,
        "npml": 20,
        "pmlR": 1e-5,
        "gain": 0,
        # PML in 0:z-, 1:x-, 2: z- and x-direction, 3: free surface
        "pml_dir": 2,

        # 0: vp, vs, rho; 1: lambda, mu, rho; 2: PCS
        "param_type": 0,

        # Number of check points relative to all time samples based on percentage
        "chpR": 15,

        "ITER_intensity": 0,
        "iteration": np.array([40, 40, 40], dtype=np.int),
        # Dominant frequency for wavelet
        "fdom": 20,
        "f_inv": np.array([15, 25, 30], dtype=np.float32),
        
        # Choosing the order of spatial derivative (Could be 2, 4, 8)
        "sdo": sdo,

        # Specify the acquisition type (0: crosswell, 1: surface, 2: both)
        "acq_type": 2,
        "energy_balancing": True,
        "gradient_smoothing": 0,

        "offset_weighting": False,

        "vel_unit": "m/s",
        "dh": dh,
        "dt": dt
    }

    for key, value in kwargs.items():
        inpa[key] = value

    return inpa

def inpa_loading(path):
    """
    inpa_loading lad the INPA file

    [extended_summary]

    Args:
        path ([type]): [description]

    Returns:
        inpa (dict): input of FWI program 
    """

    if os.path.isfile(path):
        inpa = load_mat(path)

    elif os.path.isdir(path):
        if path[-1] != "/":
            path += "/"
        inpa = load_mat(path + "INPA.mat")
    
    inpa['nx'] = inpa['nx'].item()
    inpa['nz'] = inpa['nz'].item()
    inpa['Npml'] = inpa['Npml'].item()
    inpa["gradient_smoothing"] = inpa["gradient_smoothing"].item()
    inpa['pmlR'] = inpa['pmlR'].item()
    inpa['pml_dir'] = inpa['pml_dir'].item()
    inpa["param_type"] = inpa["param_type"].item()
    inpa['chpR'] = inpa['chpR'].item()
    inpa['dh'] = inpa['dh'].item()
    inpa['dt'] = inpa['dt'].item()
    inpa['nt'] = inpa['nt'].item()
    inpa['fdom'] = inpa['fdom'].item()
    inpa["cost_function_type"] = inpa["cost_function_type"].item()
    inpa["sdo"] = inpa["sdo"].item()
    inpa["ns"] = inpa["ns"].item()
    inpa['offsetx'] = inpa['offsetx'].item()
    inpa['offsetz'] = inpa['offsetz'].item()
    inpa['f_inv'] = inpa['f_inv'].reshape(-1)
    inpa['iteration'] = inpa['iteration'].item()
    inpa['TL_inversion_method'] = inpa['TL_inversion_method'].item()

    inpa['regularization'] = {
    'tv': {
        'az': inpa['regularization'][0][0][0][0][0][0].item(),
        'ax': inpa['regularization'][0][0][0][0][0][1].item(),
        'lambda_weight':  inpa['regularization'][0][0][0][0][0][2].item(),
        'iteration_number': inpa['regularization'][0][0][0][0][0][3].item(),
        'az_tl': inpa['regularization'][0][0][0][0][0][4].item(),
        'ax_tl': inpa['regularization'][0][0][0][0][0][5].item(),
        'lambda_weight_tl': inpa['regularization'][0][0][0][0][0][6].item(),
        'iteration_number_tl': inpa['regularization'][0][0][0][0][0][7].item(),

    },
    'tikhonov': {
        'az': inpa['regularization'][0][0][1][0][0][0].item(),
        'ax': inpa['regularization'][0][0][1][0][0][1].item(),
        'lambda_weight': inpa['regularization'][0][0][1][0][0][2].item(),
        'iteration_number': inpa['regularization'][0][0][1][0][0][3].item(),
        'az_tl': inpa['regularization'][0][0][1][0][0][4].item(),
        'ax_tl': inpa['regularization'][0][0][1][0][0][5].item(),
        'lambda_weight_tl': inpa['regularization'][0][0][1][0][0][6].item(),
    },
    'tikhonov0': {
        'lambda_weight': inpa['regularization'][0][0][2][0][0][0].item()
    }
}

    return inpa


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
            model['vp'] = 1000 * rp.p_velocity().Han(model['phi'], model['cc'])
            logging.info("P-wave velocity is estimated based on Han method")

        except:
            raise Exception ("Model has to have P-wave velocity")

    if len_keys < 3:
        raise "For Elastic case (med_type=1), vp, vs, and density have to be provided."

    
    return model


def modeling_model(model, med_type):

    if med_type in [0, 'acoustic']:
        model = _acoustic_model_preparation(model, med_type)

    elif med_type in [1, 'elastic']:
       model = _elastic_model_preparation(model, med_type)


    return model


def grad_lmd_to_vd(glam, gmu, grho, lam, mu, rho):
    """
    grad_lmr_to_vd [summary]

    [extended_summary]

    Args:
        glam ([type]): [description]
        gmu ([type]): [description]
        grho ([type]): [description]
        lam ([type]): [description]
        mu ([type]): [description]
        rho ([type]): [description]
    
    Refrences:
         1. Hu et al, 2021, Direct updating of rock-physics properties using elastice full-waveform inversion
         2. Zhou and Lumely, 2021, Central-difference time-lapse 4D seismic full-waveform inversion
    """
    vp = np.sqrt((lam + 2 * mu) / rho)
    vs = np.sqrt(mu / rho)
    
    glam_vp = glam * 2 * vp * rho
    gmu_vp = gmu * rho * vp
    grho_vp = grho * (- 2 * (lam + 2*mu)/vp**3)
    gvp = glam_vp + gmu_vp + grho_vp  

    glam_vs = glam * 0
    gmu_vs = gmu * 2 * vs * rho
    
    # To not get ZeroDivisionError in acoustic case
    if np.all(vs==0):
        grho_vs  = 0
    else:
        grho_vs = grho * (-2*mu/(vs ** 3))

    gvs = glam_vs + gmu_vs + grho_vs  # gvs

    glam_rho = glam * vp ** 2
    gmu_rho = gmu * 0.5 * vp ** 2
    grho_rho = grho
    grho = glam_rho + gmu_rho + grho_rho

    return gvp, gvs, grho


def adj_grad_lmd_to_vd(gvp, gvs, grho, lam, mu, rho):
    """
    grad_lmr_to_vd [summary]

    [extended_summary]

    Args:
        glam ([type]): [description]
        gmu ([type]): [description]
        grho ([type]): [description]
        lam ([type]): [description]
        mu ([type]): [description]
        rho ([type]): [description]
    
    Refrences:
         1. Hu et al, 2021, Direct updating of rock-physics properties using elastice full-waveform inversion
         2. Zhou and Lumely, 2021, Central-difference time-lapse 4D seismic full-waveform inversion
    """
    vp = np.sqrt((lam + 2 * mu) / rho)
    vs = np.sqrt(mu / rho)
    
    gvp_lam = gvp / (2 * np.sqrt(rho * ( lam + 2 * mu)))
    gvs_lam = gvs * 0
    grho_lam = grho / (vp * vp)
    glam = gvp_lam + gvs_lam + grho_lam  # glam

    gvp_mu = gvp * (1/np.sqrt(rho * (lam + 2 * mu)))
    gvs_mu = gvs / (2 * np.sqrt(mu * rho))
    grho_mu = grho * 2 / (vp * vp)
    gmu = gvp_mu + gvs_mu + grho_mu  # gmu

    grho_vp = gvp * (- ((lam + 2 * mu) ** .5)/ (2 * rho ** 1.5))
    grho_vs = gvs * (- mu**0.5 / rho**1.5)
    grho_rho = grho
    grho = grho_vp + grho_vs + grho_rho

    return glam.astype(np.float32), gmu.astype(np.float32), grho.astype(np.float32)


def grad_vd_to_pcs(gvp0, gvs0, grho0, cc, phi, sw):
    """
    grad_vd_to_pcs [summary]

    [extended_summary]

    Args:
        gvp ([type]): [description]
        gvs ([type]): [description]
        grho ([type]): [description]
        cc ([type]): [description]
        rho_c ([type]): [description]
        rho_q ([type]): [description]
        phi ([type]): [description]
        rho_w ([type]): [description]
        rho_g ([type]): [description]
        rho_f ([type]): [description]

    Returns:
        [type]: [description]

    Refrences:
         1. Hu et al, 2021, Direct updating of rock-physics properties using elastice full-waveform inversion
         2. Zhou and Lumely, 2021, Central-difference time-lapse 4D seismic full-waveform inversion
    """
    rho_q = 2.65
    rho_c = 2.55
    rho_w = 1.0
    rho_g = 0.1

    rho_f = rp.Density().fluid(rho_g, rho_w, sw)

    gvp = np.copy(gvp0)
    gvs = np.copy(gvs0)
    grho = np.copy(grho0)

    rho_m = rp.Density().matrix(rho_c, cc, rho_q)

    gvp_phi = gvp * (-6.94 * 1000)
    gvs_phi = gvs * (- 4.94 * 1000)
    grho_phi = grho * (- rho_m + rho_f)
    gphi = gvp_phi + gvs_phi + grho_phi  # gvp

    gvp_cc = gvp * (-1728/2 /np.sqrt(cc))
    gvs_cc = gvs * (-1570/2 /np.sqrt(cc))
    grho_cc = grho * (1 - phi) * (rho_c - rho_q)
    gcc = gvp_cc + gvs_cc + grho_cc  # gvs

    gvp_s = gvp * 0
    gvs_s = gvs * 0
    grho_s = grho * phi * (rho_w - rho_g)
    grho = gvp_s + gvs_s + grho_s

    return gphi, gcc, grho

class recorder:
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

    Extended Summary
    ----------------
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

        Extended Summary
        ----------------
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
    m = {
        'vp': m0[:nz * nx].reshape(nz, nx),
        'vs': m0[nz * nx:2*nz * nx].reshape(nz, nx),
        'rho': m0[2*nz * nx:].reshape(nz, nx)
    }

    return m

def svd_reconstruction(m, begining_component, num_components):
    U, s, V = np.linalg.svd(m) 
    reconst_img = np.matrix(U[:, begining_component:begining_component +num_components]) *\
        np.diag(s[begining_component:begining_component +num_components]) * \
            np.matrix(V[begining_component:begining_component +num_components, :])
    return reconst_img


def cost_preparation(dpre, dobs,
                     fn, freq=False, order=None, axis=None,
                     params_oh=None):

    x_pre = copy.deepcopy(dpre)
    x_obs = copy.deepcopy(dobs)

    if freq:
        # highcut = freq / fn
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
         fn=None, freq=False, order=None, axis=None,
         sourc_weight=False, ns=None, nr=None,
         params_oh=None):

    d_pre = copy.deepcopy(d_pre0)
    d_obs = copy.deepcopy(d_obs0)

    dpre = np.array(list(d_pre.values()))
    dobs = np.array(list(d_obs.values()))

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
    adj_src['vx'] = adj_src_ndarray[0, :, :]
    adj_src['vz'] = adj_src_ndarray[1, :, :]
    adj_src['taux'] = adj_src_ndarray[2, :, :]
    adj_src['tauz'] = adj_src_ndarray[3, :, :]
    adj_src['tauxz'] = adj_src_ndarray[4, :, :]

    return rms, adj_src


class CostFunction:
    def __init__(self, cost_function_type="l2"):
        self.cost_function_method = "self." + cost_function_type

    def __call__(self, dest, dobs):
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
        rms = np.sum(np.abs(res))
        adj_src = np.ones(res.shape, np.float32)

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

        # adj_src = res
        
        if type(dest0).__name__ == 'ndarray':
            adj_src = np.array(list(res.values()))
            
        return rms, adj_src

    def l2_intensity(self, dest, dobs):
        # res = [dest[i] - dobs[i] for i in range(len(dest))]
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

        tau = 1500  # 6.5  #TODO make it baed on the value of l2

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


if __name__ == "__main__":
    R = recorder(['vx', 'vz'], 10, 10, 1)
    print(R.vx)