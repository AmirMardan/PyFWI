import os 

try:
    from PyFWI.seismic_io import loadmat
except:
    from seismic_io import loadmat


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
        inpa = loadmat(path)

    elif os.path.isdir(path):
        if path[-1] != "/":
            path += "/"
        inpa = loadmat(path + "INPA.mat")
    
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

