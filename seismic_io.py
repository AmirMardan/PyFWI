import numpy as np
import segyio
import gzip
import os
from urllib.request import urlretrieve
import shutil
import requests
import matplotlib.pyplot as plt


def read_segy(path):
    """
    A function to load segy file

    :param path: The path of segy file.
    :return: data
    """
    with segyio.open(path, "r", strict=False) as segy:
        models = np.transpose(np.array([segy.trace[trid]
                                        for trid in range(segy.tracecount)]))
    data = models.astype(np.float32)

    return data

