import os 
import numpy as np

try:
    from PyFWI.seismic_io import load_mat
    from PyFWI import rock_physics as RP 
except:
    from seismic_io import load_mat
    import rock_physics as RP 


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
    if len_keys == 1:
        model['rho'] = np.ones(shape, np.float32)
            
        # if keys[0] == 'lam':
        #     model['vp'] =  RP.p_velocity().lam_mu_rho(model['lam'], model['vs'], model['rho'])
    
    elif len_keys == 2:
        if 'rho' not in keys:
            model['rho'] = np.ones(shape, np.float32)
            print("Density is considered constant.")

    if keys[0] == 'lam':
        model['vp'] = RP.p_velocity().lam_mu_rho(model['lam'], model['vs'], model['rho'])

    return model
            

def _elastic_model_preparation(model, med_type):
    keys = [*model]

    len_keys = len(keys)
    shape = model[[*model][0]].shape

    if len_keys < 3:
        raise "For Elastic case (med_type=1), vp, vs, and density have to be provided."

    
    return model


def prepare_model(model, med_type):

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
    grho_vs = grho * (-2*mu/vs ** 3)
    gvs = glam_vs + gmu_vs + grho_vs  # gvs

    glam_rho = glam * vp ** 2
    gmu_rho = gmu * vs ** 2
    grho_rho = grho
    grho = glam_rho + gmu_rho + grho_rho

    return gvp, gvs, grho

def grad_vd_to_pcs(gvp, gvs, grho, mu, lam, vp, vs, rho):
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
    gvp_phi = gvp * (-6.94)
    gvs_phi = gvs * (- 4.94)
    grho_phi = grho * (- 2 * (lam - 2*mu)/vp**3)
    gvp = glam_vp + gmu_vp + grho_vp  # gvp

    glam_vs = glam * 0
    gmu_vs = gmu * 2 * vs * rho
    grho_vs = grho * (-2*mu/vs ** 3)
    gvs = glam_vs + gmu_vs + grho_vs  # gvs

    glam_rho = glam * vp ** 2
    gmu_rho = gmu * vs ** 2
    grho_rho = grho
    grho = glam_rho + gmu_rho + grho_rho

    return gvp, gvs, grho