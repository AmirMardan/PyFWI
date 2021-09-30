import numpy as np
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


def save_mat(path, unique=None, **kwarg):
    """This function save python dictionary as a .mat file.

    Parameters
    ----------
    path : String
        The path to save the data.
    unique : Boolean
        If true, it will add current date and time to the name of folder
    **kwarg : type
        Dictionaries containing the data.

    """

    if unique:
        path += datetime.datetime.now().strftime("_%b_%d_%Y_%H_%M/")

    try:
        os.makedirs(path)
    except:
        pass
    for params in kwarg:
        path_case = path + params + ".mat"
        sio.savemat(path_case, kwarg[params], oned_as='row')


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
