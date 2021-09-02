import numpy as np
import segyio
import gzip
import os
from urllib.request import urlretrieve
import shutil
import requests


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

