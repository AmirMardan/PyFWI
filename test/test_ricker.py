"""
To test wavlets
"""
import sys
import numpy as np
import os
pyfwi_path = os.path.abspath(os.path.join(__file__, "../../src"))
sys.path.append(pyfwi_path)
from PyFWI.acquisition import Source

def ricker(fdom:np.float32, dt:np.float32):
    src = Source(src_loc=np.array([[0,0]]), dh=1.0, dt=dt)
    src.Ricker(fdom=fdom)
    return src.w

def test_ricker():
    fdom = 20.0
    dt = 0.004
    w = ricker(fdom=fdom, dt=dt)
    t = np.arange(-1.0/fdom, 1.0/fdom + dt/3, dt)
    assert np.all(w == np.float32((1.0 - 2.0*(np.pi * fdom * t) **2 ) * \
            np.exp(-(np.pi * fdom * t) ** 2)))
    