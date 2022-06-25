import numpy as np
from numpy.lib.function_base import kaiser
import segyio
import os
from urllib.request import urlretrieve
import scipy.io as sio
import datetime
import pickle
import hdf5storage as hdf5


def read_segy(path):
    """ A function to load segy file.

    Args:
        path: The path of segy file.

    Returns:
        data: The data stored in segy.
    """
    with segyio.open(path, "r", strict=False) as segy:
        models = np.transpose(np.array([segy.trace[trid]
                                        for trid in range(segy.tracecount)]))
    data = models.astype(np.float32)

    return data


def save_mat(path, **kwargs):
    """This function save python dictionary as a .mat file.

    Parameters
    ----------
    path : String
        The path to save the data.
    unique : Boolean
        If true, it will add current date and time to the name of folder
    **kwargs : type
        Dictionaries containing the data.

    """
    try:
        if path[-1] != "/":
            path += "/"
    except:
        pass
        
    keys = kwargs.keys()
    
    if "unique" in keys:
        if kwargs["unique"] == True:
            path += datetime.datetime.now().strftime("%b_%d_%Y_%H_%M/")
        kwargs.pop("unique")

    try:
        os.makedirs(path)
    except:
        pass
    for params in kwargs:
        path_case = path + params + ".mat"
        sio.savemat(path_case, kwargs[params], oned_as='row')


def load_mat(path):
    """This function load python dictionary as a .mat file.

    Parameters
    ----------
    path : String
        The path to save the data.
    """
    data = {}
    hdf5.loadmat(path, data)
    
    try:
        data.pop("__header__")
        data.pop("__version__")
        data.pop("__globals__")
    except:
        pass
    
    return data

#%% pkl
def save_pkl(path, **kwargs):
    """
    save_pkl saves pkl file.

    save_pkl saves file with pkl format. That is better than .mat file
    for preserving the structure of dictionaries. 

    Args:
        path (string): path to save the file(s).

        **kwargs (data): Variable(s) to be saved.
        A boolean argument with name of "unique" can be given to make the 
        path based on the data. 

    """
    try:
        if path[-1] != "/":
            path += "/"
    except:
        pass

    keys = kwargs.keys()
    if "unique" in keys:
        if kwargs["unique"] == True:
            path += datetime.datetime.now().strftime("%b_%d_%Y_%H_%M/")
        kwargs.pop("unique")

    try:
        os.makedirs(path)
    except:
        pass

    for params in kwargs:
        path_case = path + params + ".pkl"

        a_file = open(path_case, "wb")
        pickle.dump(kwargs[params], a_file)
        a_file.close()


def load_pkl(file_path):
    """
    load_pkl loads pkl file.

    load_pkl loads pkl file.

    Args:
        file_path (string): Path of file to be loaded.

    Returns:
        output: Loaded file.
    """
    a_file = open(file_path, "rb")
    output = pickle.load(a_file)
    return output

if __name__ == "__main__":
    test = {'vp':1,
            'vs':{
                'k':1,
                'c':1
            }}

    save_mat("", test=test, unique=False)