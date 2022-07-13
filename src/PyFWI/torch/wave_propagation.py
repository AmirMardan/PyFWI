import os
import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as F

import PyFWI.fwi_tools as tools
import PyFWI.acquisition as acq
from PyFWI.processing import prepare_residual
from PyFWI.grad_swithcher import grad_lmd_to_vd

class Fdm:
    def __init__(self, order):
        if order == 4:
            self._c1 = 9/8
            self._c2 = - 1 / 24
            self._c3 = 0
            self._c4 = 0

        elif order == 8:
            self._c1 = 1715 / 1434
            self._c2 = -114 / 1434
            self._c3 = 14 / 1434
            self._c4 = -1 / 1434
        else:
            raise AssertionError ("Order of the derivative has be either 4 or 8!")
    
    def dzp(self, *args):
        return self.dzp4(*args)
    
    def dzm(self, *args):
        return self.dzm4(*args)
    
    def dxp(self, *args):
        return self.dxp4(*args)
    
    def dxm(self, *args):
        return self.dxm4(*args)
    
    def dxm4(self, x, dx):
        y = (self._c1 * (x[2:-2, 2:-2] - x[2:-2, 1:-3]) +
                         self._c2 * (x[2:-2, 3:-1] - x[2:-2, :-4])) / dx
        y = F.pad(input=y, pad=(2, 2, 2, 2), mode='constant', value=0)
        return y
    
    def dxp4(self, x, dx):

        y = (self._c1 * (x[2:-2, 3:-1] - x[2:-2, 2:-2]) +
                         self._c2 * (x[2:-2, 4:] - x[2:-2, 1:-3])) / dx
        y = F.pad(input=y, pad=(2, 2, 2, 2), mode='constant', value=0)
        return y
    
    def dzm4(self, x, dx):

        y = (self._c1 * (x[2:-2, 2:-2] - x[1:-3, 2:-2]) +
                         self._c2 * (x[3:-1, 2:-2] - x[:-4, 2:-2])) / dx
        y = F.pad(input=y, pad=(2, 2, 2, 2), mode='constant', value=0)
        return y 
    
    def dzp4(self, x, dx):

        y = (self._c1 * (x[3:-1, 2:-2] - x[2:-2, 2:-2]) +
                         self._c2 * (x[4:, 2:-2] - x[1:-3, 2:-2])) / dx
        y = F.pad(input=y, pad=(2, 2, 2, 2), mode='constant', value=0)
        return y


class WavePreparation:

    def __init__(self, inpa, src, rec_loc, model_shape, n_well_rec=0, chpr=10, components=0, device='cpu'):
        '''
        A class to prepare the variable and basic functions for wave propagation.

        '''
        #TODO: work on how ypu specify the acq_type, getting n_well_rec, using that again fpr two .cl files
        keys = [*inpa]

        self.device = device
        
        self.t = inpa['t']
        self.dt = inpa['dt']
        self.nt = int(1 + self.t // self.dt)

        self.nx = np.int32(model_shape[1])
        self.nz = np.int32(model_shape[0])

        if 'g_smooth' in keys:
            self.g_smooth = inpa['g_smooth']
        else:
            self.g_smooth = 0

        if 'npml' in keys:
            self.npml = inpa['npml']
            pmlR = inpa['pmlR']
            pml_dir = inpa['pml_dir']
        else:
            self.npml = 0
            pmlR = 0
            pml_dir = 2
                
        # Number of samples in x- and z- direction by considering pml
        self.tnx = np.int32(self.nx + 2 * self.npml)
        self.tnz = np.int32(self.nz + 2 * self.npml)

        self.dh = np.float32(inpa['dh'])

        self.srcx = np.int32(src.i + self.npml)
        self.srcz = np.int32(src.j + self.npml)
        src_loc = np.vstack((self.srcx, self.srcz)).T

        self.src = src
        self.ns = np.int32(src.i.size)
        
        self.rec_loc = rec_loc
        self.nr = rec_loc.shape[0]
    
        # ======== Parameters Boundary condition ======
        self.dx_pml, self.dz_pml = tools.pml_counstruction(self.tnz, self.tnx, self.dh, self.npml,
                                                     pmlR, pml_dir)

        self.components = components

        # TODO: Can be just for pressure
        # Make a list for seismograms
        self.seismogram = {
            'vx': torch.zeros((self.nt, self.nr, self.ns)),
            'vz': torch.zeros((self.nt, self.nr, self.ns)),
            'p': torch.zeros((self.nt, self.nr, self.ns)),
        }
        
        #TODO TO CHANGE to acoustic
    def acoustc_update_v(self, vx_b, vz_b, px_b, pz_b):
        vx_b += - self.dt * self.vdx_pml_b * vx_b + \
            self.dt * (self.dxp(px_b, self.dh) + self.dxp(pz_b, self.dh))
        vz_b += - self.dt * self.vdz_pml_b * vz_b + \
            self.dt * (self.dzp(px_b, self.dh) + self.dzp(pz_b, self.dh))

    def acoustic_update_tau(self, vx_b, vz_b, px_b, pz_b):            
        px_b += - self.dt * self.vdx_pml_b * px_b + \
                self.dt * self.vp ** 2 * self.dxm(vx_b, self.dh) 

        pz_b += - self.dt * self.vdz_pml_b * pz_b + \
                self.dt * self.vp ** 2 * self.dzm(vz_b, self.dh)


    def make_seismogram(self, vx, vz, px, pz, s, t):
        """
        This function read the seismogram buffer and put its value in the
        right place in main seismogram matrix based on source and the
        time step.

        Parameters
        ----------
            s : int
                Number of current acive source.

            t : float
                Current time step.
        """
        rec_loc = np.int32(self.rec_loc / self.dh)
        def get_from_gpu(buffer):
            return buffer[rec_loc[:, 1], rec_loc[:, 0]]

        self.seismogram['vx'][np.int32(t - 1), :, s] = \
            get_from_gpu(vx)

        self.seismogram['vz'][np.int32(t - 1), :, s] = \
            get_from_gpu(vz)

        self.seismogram['p'][np.int32(t - 1), :, s] = \
            (get_from_gpu(px) + get_from_gpu(pz)) / 2

    def pml_preparation(self, v_max):

        self.vdx_pml_b = torch.tensor(self.dx_pml, device=self.device) * v_max
        self.vdz_pml_b = torch.tensor(self.dz_pml, device=self.device) * v_max

    def initial_wavefield_plot(self, model, plot_type="Forward"):
        """
        A function to initialize the the plot for wave
        propagation visulizing

        Parameters
        ----------
            plot_type: string optional = "Forward"
                Specify if we want to show Forward modelloing or Backward.

        """
        key = [*model][0]
        show_purpose = np.zeros((self.tnz - 2 * self.npml, self.tnx - 2 * self.npml))
        if plot_type == "Forward":
            fig, ax = plt.subplots(1, 1)
            self.__im0 = ax.imshow(model[key].detach()[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml], cmap='jet',
                                   vmin=show_purpose.min())
            fig.colorbar(self.__im0, extend='both', label=key, shrink=0.3)
            self.__im0 = ax.imshow(show_purpose, alpha=.65, cmap="seismic")

        elif plot_type == "Backward":
            fig, ax = plt.subplots(1, 2)
            self.__im1 = ax[1].imshow(show_purpose)
            self.__im0 = ax[0].imshow(show_purpose)
        self.__stitle = fig.suptitle('')

    def plot_propagation(self, wave1, t, wave2=None):
        """
        This function is used to shpw the propagation wave with time.

        Parameters
        ----------

        wave1 : float32
            The wave that we are going to show

        t : float32
            Time step

        wave2 : flaot32, optional = None
            The second wave which is used when we want to show the propagation
            of backward wave as wave1 and adjoint wave as wave2.

        """

        wave1 = wave1[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml]

        self.__im0.set_data(wave1)  # [self.Npml:self.TNz - self.Npml, self.Npml:self.TNx - self.Npml])
        self.__im0.set_clim(wave1.min(), wave1.max())

        if wave2 is not None:
            wave2 = wave2[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml]
            self.__im1.set_data(wave2)
            self.__im1.set_clim(wave2.min() / 20, wave2.max() / 20)
        self.__stitle.set_text(
            't = {0:6.3f} (st. no {1:d}/{2:d})'.format(t * self.dt, t + 1, self.nt))
        plt.pause(0.1)


    def elastic_buffers(self, model):
        '''
        Model hast contain vp, vs, and rho
        '''
        self.vp = model['vp']


class WavePropagator(WavePreparation, Fdm):
    """
    wave_propagator is a class to handle the forward modeling and gradient calculation.

    [extended_summary]

    Parameters
    ----------
    inpa : dict
        A dictionary including most of the required inputs
    src : class
        Source object
    rec_loc : ndarray
        Location of the receivers
    model_shape : tuple
        Shape of the model
    n_well_rec: int
        Number of receivers in the well
    chpr : percentage
        Checkpoint ratio in percentage
    component:
        Seismic output
    """
    def __init__(self, inpa, src, rec_loc, model_shape, n_well_rec=None, chpr=0, components=0, device='cpu'):
        WavePreparation.__init__(self, inpa, src, rec_loc, model_shape, n_well_rec, chpr=chpr,
                                 components=components, device=device)
        
        if 'sdo' in inpa:
            sdo = np.int32(inpa['sdo'] / 2)
        else:
            sdo = 4
            
        Fdm.__init__(self, 2 * sdo)
        
    def forward_propagator(self, model):
        """ This function is in charge of forward modelling for acoustic case

        Parameters
        ---------
            model: dictionary
                A dictionary containing p-wave velocity and density

        """

        if self.forward_show:
            self.initial_wavefield_plot(model)

        for s in range(self.ns):

            self.__kernel(s)

        return self.seismogram

    def __kernel(self, s):
        showpurose = np.zeros((self.tnz, self.tnx), dtype=np.float32)
        
        vx_b = torch.zeros((self.tnz, self.tnx), device=self.device)
        vz_b = torch.zeros((self.tnz, self.tnx), device=self.device)
        taux_b = torch.zeros((self.tnz, self.tnx), device=self.device)
        tauz_b = torch.zeros((self.tnz, self.tnx), device=self.device)

        
        for t in range(self.nt):
            src_v_x, src_v_z, src_t_x, src_t_z, src_t_xz = np.float32(self.src(t))

            inj_src(vx_b, vz_b, taux_b, tauz_b,
                   self.srcx[s], self.srcz[s],
                   src_v_x, src_v_z,src_t_x, src_t_z)

            self.acoustc_update_v(vx_b, vz_b, taux_b, tauz_b)

            self.acoustic_update_tau(vx_b, vz_b, taux_b, tauz_b)
            
            self.make_seismogram(vx_b, vz_b, taux_b, tauz_b, s, t)

            if self.forward_show and np.remainder(t, 20) == 0:
                showpurose = np.copy(taux_b.detach().numpy())
                self.plot_propagation(showpurose, t)

    def forward_modeling(self, model0, show=False):
        """
        forward_modeling performs the forward modeling.


        Parameters
        ----------
        model0 : dict
            The earth model
        show : bool, optional
            True if you desire to see the propagation of the wave, by default False

        Returns
        -------
        dict
            Seismic section
        """
        self.forward_show = show
        model = model0.copy()

        for params in model:
            # model[params] = model[params]  # To avoid sticking BC. to the original model
            model[params] = expand_model(model[params], self.tnz, self.tnx, self.npml)

        self.pml_preparation(model['vp'].max().item())
        self.elastic_buffers(model)
        seismo = self.forward_propagator(model)
        
        # data = acq.seismic_section(seismo, self.components, shape='3d')
        return {'p': seismo['p']}


def expand_model(parameter, TNz, TNx, n_pml=10, tensor='torch'):
    """
    This function make room around the 'parameter' to stick the pml layer.

    Parameters
    ----------
        parameter : float
            Matrix of property that we are going to consider pml around.

        TNz : int
            Number of samples in z-direction (`n_z + 2 * n_pml`).

        TNx : int
            Number of samples in x-direction (`n_x + 2 * n_pml`).

        n_pml : int, optional = 10
            Number of pml layer

    Returns
    --------
        nu : float
            A matrix with the size of [TNz, TNx] with zero value
            everywhere excpet in center which consisting the model.

    """
    if tensor == 'torch':
        nu = torch.nn.functional.pad(parameter, (n_pml, n_pml, n_pml, n_pml), value=0.0)
        
    nu[:n_pml, :] = nu[n_pml, :]
    nu[TNz - n_pml:, :] = nu[TNz - n_pml - 1, :]

    nu[:, TNx - n_pml:] = nu[:, TNx - n_pml - 1].reshape(-1, 1)
    nu[:, :n_pml] = nu[:, n_pml].reshape(-1, 1)
    return nu


def inj_src(vx_b, vz_b, taux_b, tauz_b,
           srcx, srcz,
           src_v_x, src_v_z, src_t_x, src_t_z):
        
    vx_b[srcz, srcx] += torch.tensor(src_v_x)
    vz_b[srcz, srcx] += torch.tensor(src_v_z)
    taux_b[srcz, srcx] += torch.tensor(src_t_x)
    tauz_b[srcz, srcx] += torch.tensor(src_t_z)




if __name__ == "__main__":
    import PyFWI.model_dataset as md
    import PyFWI.acquisition as acq
    import PyFWI.seiplot as splt
    import PyFWI.fwi as fwi
    
    GRADIENT = 1
    INVERSION = 0

    model_gen = md.ModelGenerator('louboutin') #

    model = model_gen()
    model['vs'] *=0
    # model_gen.show(['vs'])
    model_shape = model[[*model][0]].shape

    inpa = {}
    # Number of pml layers
    inpa['npml'] = 20
    inpa['pmlR'] = 1e-5
    inpa['pml_dir'] = 2
    # inpa['device'] = 0
    inpa['energy_balancing'] = False
    inpa['seisout'] = 0

    chpr = 100
    sdo = 4
    fdom = 25
    inpa['fn'] = 125
    vp = model['vp']
    D = tools.Fdm(order=sdo)
    dh = vp.min()/(D.dh_n * inpa['fn'])
    dh = 2.
    inpa['dh'] = dh

    dt = D.dt_computation(vp.max(), inpa['dh'])
    inpa['dt'] = dt

    # print(f'{dh = } ........... {dt = }')
    inpa['t'] = 0.36

    offsetx = inpa['dh'] * model_shape[1]
    depth = inpa['dh'] * model_shape[0]

    inpa['rec_dis'] = 2.  # inpa['dh']
    ns = 1
    inpa['acq_type'] = 0

    src_loc, rec_loc, n_surface_rec, n_well_rec = acq.acq_parameters(ns, inpa['rec_dis'], offsetx, depth, inpa['dh'], sdo, inpa['acq_type'])

    src = acq.Source(src_loc, inpa['dh'], inpa['dt'])
    src.Ricker(fdom)

    W = WavePropagator(inpa, src, rec_loc, model_shape, n_well_rec, chpr=0, components=inpa['seisout'])
    d_obs = W.forward_modeling(model, False)
    d_obs = prepare_residual(d_obs, 1)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    splt.seismic_section(ax, d_obs['taux'], vmin=d_obs['taux'].min() / 5, vmax=d_obs['taux'].max() / 5)

    m0 = model_gen(vintage=1, smoothing=True)

    if GRADIENT:
        Lam = WavePropagator(inpa, src, rec_loc, model_shape, n_well_rec, chpr=chpr, components=inpa['seisout'])
        d_est = Lam.forward_modeling(m0, show=False)
        d_est = prepare_residual(d_est, 1)

        CF = tools.CostFunction('l2')
        rms, res = CF(d_est, d_obs)

        print(rms)

        grad = Lam.gradient(res, show=False)

        splt.earth_model(grad, cmap='jet')
        plt.show()
        a=1

    elif INVERSION:

        FWI = fwi.FWI(d_obs, inpa, src, rec_loc, model_shape, n_well_rec, chpr=chpr, components=4, param_functions=None)
        inverted_model, rms = FWI(m0, method=2, iter=[3], freqs=[25], n_params=1, k_0=1, k_end=2)


        splt.earth_model(inverted_model, cmap='jet')
        plt.show(block=False)

        plt.figure()
        plt.plot(rms/max(rms))
        plt.show()
