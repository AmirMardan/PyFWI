import numpy as np
from numpy.lib.function_base import kaiser
import segyio
import gzip
import os
from urllib.request import urlretrieve
import shutil
import requests
import scipy.io as sio
import datetime


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


def savemat(path, **kwargs):
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
    if path in ["", "/"] :
        path = os.getcwd() + "/"

    if path[-1] != '/':
        path += '/'
        
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


def loadmat(path):
    """This function load python dictionary as a .mat file.

    Parameters
    ----------
    path : String
        The path to save the data.
    """

    data = sio.loadmat(path)

    try:
        data.pop("__header__")
        data.pop("__version__")
        data.pop("__globals__")
    except:
        pass

    return data
