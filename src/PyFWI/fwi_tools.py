import os 
import numpy as np
import copy 
import logging

from numpy.core.arrayprint import dtype_is_implied 
try:
    from PyFWI.seismic_io import load_mat
    from PyFWI import rock_physics as rp 
except:
    from seismic_io import load_mat
    import rock_physics as rp 


def inpa_generator(**kwargs):
    inpa = {
        "SeisCL": False,
        "seisout": 4,
        "no_use_GPUs": np.array([-1]),
        "cost_function_type": "l2",
        "cost_function_intensity": "l2_intensity",
        "device": 0,
        "medium": 1,
        "ns": 1,
        "Npml": 20,
        "pmlR": 1e-5,
        "gain": 0,
        # PML in 0:z-, 1:x-, 2: z- and x-direction, 3: free surface
        "pml_dir": 2,

        # 0: vp, vs, rho; 1: lambda, mu, rho; 2: PCS
        "param_type": 0,

        # Number of check points relative to all time samples based on percentage
        "chpR": 15,

        # Defining the data type (0: circle, 1: Simple two layers, 2: Dupuy, 3: Marmousi,
        #                         4: st Lawrence, 5: perturbation, 6: SEAM, 7:Hu1)
        "models_name": 5,

        "ITER_intensity": 0,
        "iteration": np.array([40, 40, 40], dtype=np.int),
        # Dominant frequency for wavelet
        "fdom": 20,
        "f_inv": np.array([15, 25, 30], dtype=np.float32),
        
        # Choosing the order of spatial derivative (Could be 2, 4, 8)
        "sdo": 8,

        # Specify the acquisition type (0: crosswell, 1: surface, 2: both)
        "acq_type": 2,
        "energy_balancing": True,
        "gradient_smoothing": 0,

        "offset_weighting": False,

        "vel_unit": "m/s",
        "dh": np.float32(7),
        "dt": 0.00061
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


# def _model_rearanging(model0, med_type):
#     model = {}
    
#     if med_type == 1:
#         try:




def grad_lmr_to_vd(glam, gmu, grho, mu, lam, vp, vs, rho):
    """
    grad_lmr_to_vd [summary]

    [extended_summary]

    Args:
        glam ([type]): [description]
        gmu ([type]): [description]
        grho ([type]): [description]
        mu ([type]): [description]
        lam ([type]): [description]
        vp ([type]): [description]
        vs ([type]): [description]
        rho ([type]): [description]
    
    Refrences:
         1. Hu et al, 2021, Direct updating of rock-physics properties using elastice full-waveform inversion
         2. Zhou and Lumely, 2021, Central-difference time-lapse 4D seismic full-waveform inversion
    """
    glam_vp = glam * 2 * vp * rho
    gmu_vp = gmu * (- rho * vp)
    grho_vp = grho * (- 2 * (lam - 2*mu)/vp**3)
    gvp = glam_vp + gmu_vp + grho_vp  # gvp

    glam_vs = glam * 0
    gmu_vs = gmu * 2 * vs * rho
    
    # vs is zeros for acoustic case
    if np.all(vs==0):
        grho_vs  = 0
    else:
        grho_vs = grho * (-2*mu/vs ** 3)

    gvs = glam_vs + gmu_vs + grho_vs  # gvs

    glam_rho = glam * vp ** 2
    gmu_rho = gmu * vs ** 2
    grho_rho = grho
    grho = glam_rho + gmu_rho + grho_rho

    return gvp, gvs, grho

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
                
        self.rec_loc = np.int32(rec_loc//dh)
        self.nr = rec_loc.shape[0]
            
        self.vx = np.zeros((nt, ns * self.nr))
        self.vz = np.zeros((nt, ns * self.nr))
        self.taux = np.zeros((nt, ns * self.nr))
        self.tauz = np.zeros((nt, ns * self.nr))
        self.tauxz = np.zeros((nt, ns * self.nr))
            
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
        
def cost_function(d_est, d_obs):
    
    res = [d_est[key] - d_obs[key] for key in d_obs]
    res = np.array(res).reshape(-1, 1) 
    
    rms = 0.5 * np.dot(res.T, res)
    return np.squeeze(rms)


    
if __name__ == "__main__":
    R = recorder(['vx', 'vz'], 10, 10, 1)
    print(R.vx)