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

    return inpa

