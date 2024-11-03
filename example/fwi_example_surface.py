import matplotlib.pyplot as plt
import numpy as np
import os

import sys
sys.path.append(os.path.abspath('./src/'))

import PyFWI.wave_propagation as wave
import PyFWI.acquisition as acq
import PyFWI.seiplot as splt
import PyFWI.model_dataset as md
from PyFWI.fwi import FWI

Model = md.ModelGenerator('louboutin')
model = Model()

im = splt.earth_model(model, cmap='coolwarm')

model_shape = model[[*model][0]].shape
#%% ================== Parameters and Geometry ==================

inpa = {
    'ns': 4,  # Number of sources
    'sdo': 4,  # Order of FD
    'fdom': 15,  # Central frequency of source
    'dh': 7,  # Spatial sampling rate
    'dt': 0.004,  # Temporal sampling rate
    'acq_type': 1,  # Type of acquisition (0: crosswell, 1: surface, 2: both)
    't': 0.6,  # Length of operation
    'npml': 20,  # Number of PML 
    'pmlR': 1e-5,  # Coefficient for PML (No need to change)
    'pml_dir': 2,  # type of boundary layer
}

seisout = 0 # Type of output 0: Pressure

inpa['rec_dis'] =  1 * inpa['dh']  # Define the receivers' distance


offsetx = inpa['dh'] * model_shape[1]
depth = inpa['dh'] * model_shape[0]

src_loc, rec_loc, n_surface_rec, n_well_rec = acq.acq_parameters(inpa['ns'], 
                                                                 inpa['rec_dis'], 
                                                                 offsetx,
                                                                 depth,
                                                                 inpa['dh'], 
                                                                 inpa['sdo'], 
                                                                 acq_type=inpa['acq_type'])        
rec_loc[:, 1] -= 2 * inpa['dh']

# Create the source
src = acq.Source(src_loc, inpa['dh'], inpa['dt'])
src.Ricker(inpa['fdom'])

#%% ================== Forward Modelling ==================
# Create the wave object
W = wave.WavePropagator(inpa, src, rec_loc, model_shape,
                        n_well_rec=n_well_rec,
                        components=seisout, chpr=0)

# Call the forward modelling 
d_obs = W.forward_modeling(model, show=False)  # show=True can show the propagation of the wave

plt.imshow(d_obs["taux"], cmap='gray', 
           aspect="auto", extent=[0, d_obs["taux"].shape[1], inpa['t'], 0])

#%% ================== Initial model ==================
m0 = Model(smoothing=1)
m0['vs'] *= 0.0
m0['rho'] = np.ones_like(model['rho'])

fig = splt.earth_model(m0, ['vp'], cmap='coolwarm')

fig.axes[0].plot(src_loc[:,0]//inpa["dh"], 
                 src_loc[:,1]//inpa["dh"], "rv", markersize=5)

fig.axes[0].plot(rec_loc[:,0]//inpa["dh"], 
                 rec_loc[:,1]//inpa["dh"], "b*", markersize=3)


#%% ================== FWI ==================
inpa['energy_balancing'] = True

fwi = FWI(d_obs, inpa, src, rec_loc, model_shape, 
          components=seisout, chpr=20, n_well_rec=n_well_rec)

m_est, _ = fwi(m0, method="lbfgs", 
                 freqs=[25, 45], iter=[2, 2], 
                 n_params=1, k_0=1, k_end=2)

fig = splt.earth_model(m_est, ['vp'], cmap='jet')

plt.show()
a = 1