import os
import numpy as np
import torch 
import matplotlib.pyplot as plt

import PyFWI.fwi_tools as tools
import PyFWI.acquisition as acq
from PyFWI.processing import prepare_residual
from PyFWI.grad_swithcher import grad_lmd_to_vd

class Fdm(object):
    def __init__(self, order, tensor='torch', device='cpu'):
        """
        Fdm is a class to implemenet the the finite difference method for wave modeling

        The coeeficients are based on Lavendar, 1988 and Hasym et al., 2014.

        Args:
            order (int, optional): [description]. Defaults to 4.
        """
        self._order = order
        self.__device = device 
        
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
            
        if tensor == 'torch':
            self._c1 = torch.tensor(self._c1, device=self.__device)
            self._c2 = torch.tensor(self._c2, device=self.__device)
            self._c3 = torch.tensor(self._c3, device=self.__device)
            self._c4 = torch.tensor(self._c4, device=self.__device)
        
        dh_n = { # Dablain, 1986, Bai et al., 2013
            '4': 4,
            '8': 3
        }

        self.dh_n = dh_n[str(order)] # Pick the appropriate n for calculating dh

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        if value in [4, 8]:
            self._order = value
        else:
            raise AssertionError ("Order of the derivative has be either 4 or 8!")

    @property
    def c1(self):
        return self._c1

    @c1.setter
    def c1(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")

    @property
    def c2(self):
        return self._c2

    @c2.setter
    def c2(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")

    @property
    def c3(self):
        return self._c3

    @c3.setter
    def c3(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")

    @property
    def c4(self):
        return self._c4

    @c4.setter
    def c3(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")

    def dxp(self, x, dx):
         if self.order == 4:
             return self._dxp4(x, dx)
         else:
             return self._dxp8(x, dx)

    def dxm(self, x, dx):
         if self.order == 4:
             return self._dxm4(x, dx)
         else:
             return self._dxm8(x, dx)

    def dzp(self, x, dx):
         if self.order == 4:
             return self._dzp4(x, dx)
         else:
             return self._dzp8(x, dx)

    def dzm(self, x, dx):
         if self.order == 4:
             return self._dzm4(x, dx)
         else:
             return self._dzm8(x, dx)

    def _dxp4(self, x, dx):
        y = torch.zeros(x.shape, device=self.__device)

        y[2:-2, 2:-2] = (self._c1 * (x[2:-2, 3:-1] - x[2:-2, 2:-2]) +
                         self._c2 * (x[2:-2, 4:] - x[2:-2, 1:-3])) / dx
        return y

    def _dxp8(self, x, dx):
        y = torch.zeros(x.shape, device=self.__device)

        y[4:-4, 4:-4] = (self._c1 * (x[4:-4, 5:-3] - x[4:-4, 4:-4]) +
                         self._c2 * (x[4:-4, 6:-2] - x[4:-4, 3:-5]) +
                         self._c3 * (x[4:-4, 7:-1] - x[4:-4, 2:-6]) +
                         self._c4 * (x[4:-4, 8:  ] - x[4:-4, 1:-7])) / dx
        return y


    def _dxm4(self, x, dx):
        y = torch.zeros(x.shape, device=self.__device)

        y[2:-2, 2:-2] = (self._c1 * (x[2:-2, 2:-2] - x[2:-2, 1:-3]) +
                         self._c2 * (x[2:-2, 3:-1] - x[2:-2, :-4])) / dx
        return y

    def _dxm8(self, x, dx):
        y = torch.zeros(x.shape, device=self.__device)

        y[4:-4, 4:-4] = (self._c1 * (x[4:-4, 4:-4] - x[4:-4, 3:-5]) +
                         self._c2 * (x[4:-4, 5:-3] - x[4:-4, 2:-6]) +
                         self._c3 * (x[4:-4, 6:-2] - x[4:-4, 1:-7]) +
                         self._c4 * (x[4:-4, 7:-1] - x[4:-4, :-8]) ) / dx
        return y

    def _dzp4(self, x, dx):
        y = torch.zeros(x.shape, device=self.__device)

        y[2:-2, 2:-2] = (self.c1 * (x[3:-1, 2:-2] - x[2:-2, 2:-2]) +
                         self.c2 * (x[4:, 2:-2] - x[1:-3, 2:-2])) / dx
        return y

    def _dzp8(self, x, dx):
        y = torch.zeros(x.shape, device=self.__device)

        y[4:-4, 4:-4] = (self.c1 * (x[5:-3, 4:-4] - x[4:-4, 4:-4]) +
                         self.c2 * (x[6:-2, 4:-4] - x[3:-5, 4:-4]) +
                         self.c3 * (x[7:-1, 4:-4] - x[2:-6, 4:-4]) +
                         self.c4 * (x[8: , 4:-4] - x[1:-7, 4:-4])) / dx
        return y


    def _dzm4(self, x, dx):
        y = torch.zeros(x.shape, device=self.__device)

        y[2:-2, 2:-2] = (self.c1 * (x[2:-2, 2:-2] - x[1:-3, 2:-2]) +
                         self.c2 * (x[3:-1, 2:-2] - x[:-4, 2:-2])) / dx
        return y

    def _dzm8(self, x, dx):
        y = torch.zeros(x.shape, device=self.__device)

        y[4:-4, 4:-4] = (self.c1 * (x[4:-4, 4:-4] - x[3:-5, 4:-4]) +
                         self.c2 * (x[5:-3, 4:-4] - x[2:-6, 4:-4]) +
                         self.c3 * (x[6:-2, 4:-4] - x[1:-7, 4:-4]) +
                         self.c4 * (x[7:-1, 4:-4] - x[ :-8, 4:-4])) / dx
        return y


class WavePreparation:

    def __init__(self, inpa, src, rec_loc, model_shape, n_well_rec=0, chpr=10, components=0, set_env_variable=True):
        '''
        A class to prepare the variable and basic functions for wave propagation.

        '''
        #TODO: work on how ypu specify the acq_type, getting n_well_rec, using that again fpr two .cl files
        keys = [*inpa]

        self.set_env_variable = set_env_variable
        
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
            self.pmlR = inpa['pmlR']
            self.pml_dir = inpa['pml_dir']
        else:
            self.npml = 0
            self.pmlR = 0
            self.pml_dir = 2
            
        if 'sdo' in keys:
            self.sdo = np.int32(inpa['sdo'] / 2)
        else:
            self.sdo = 2

        if 'rec_dis' in keys:
            rec_dis = inpa['rec_dis']
        else:
            rec_dis = (rec_loc[1,:] - rec_loc[0,:]).max()
        
        if 'acq_type' in keys:
            self.acq_type = inpa['acq_type']
        else:
            self.acq_type = 1
            
        if 'energy_balancing' in keys:
            self.energy_balancing = inpa['energy_balancing']
        else:
            self.energy_balancing = False
            
            
        # Number of samples in x- and z- direction by considering pml
        self.tnx = np.int32(self.nx + 2 * self.npml)
        self.tnz = np.int32(self.nz + 2 * self.npml)

        self.dh = np.float32(inpa['dh'])

        self.srcx = np.int32(src.i + self.npml)
        self.srcz = np.int32(src.j + self.npml)
        src_loc = np.vstack((self.srcx, self.srcz)).T

        self.src = src
        self.ns = np.int32(src.i.size)
        
        self.dxr = np.int32(rec_dis / self.dh)

        self.chpr = chpr
        chp = int(chpr * self.nt / 100)
        self.chp = np.linspace(0, self.nt-1, chp, dtype=np.int32)
        if (len(self.chp) < 2): #& (chpr != 0)
            self.chp = np.array([1, self.nt-1])

        self.nchp = len(self.chp)
        # Take chpr into account

        self.rec_loc = rec_loc
        self.nr = rec_loc.shape[0]
        self.n_well_rec = n_well_rec
        self.n_surface_rec = self.nr - 2 * n_well_rec

        if n_well_rec ==0 and self.acq_type == 2:
            raise Exception(" Number of geophons in the wells is not defined")

        # The matrix containg the geometry of acquisittion (Never used really)
        data_guide = acq.acquisition_plan(self.ns, self.nr, src_loc, self.rec_loc, self.acq_type, n_well_rec, self.dh)

        self.data_guide_sampling = acq.discretized_acquisition_plan(data_guide, self.dh, self.npml)

        self.rec_top_left_const = np.int32(0)
        self.rec_top_left_var = np.int32(0)
        self.rec_top_right_const = np.int32(0)
        self.rec_top_right_var = np.int32(0)
        self.rec_surface_const = np.int32(0)
        self.rec_surface_var = np.int32(0)

        if self.acq_type == 0:
            self.rec_top_right_const = np.int32(rec_loc[0, 0] / self.dh + self.npml)
            self.rec_top_right_var = np.int32(rec_loc[:, 1] / self.dh + self.npml)[0]
            self.src_cts = src.i[0]

        elif self.acq_type == 1:
            self.rec_surface_const = np.int32(rec_loc[0, 1] / self.dh + self.npml)
            self.rec_surface_var = np.int32(rec_loc[:, 0] / self.dh + self.npml)[0]
            self.src_cts = src.j[0]

        elif self.acq_type == 2:
            a = np.int32(rec_loc[-self.n_well_rec, :]/ self.dh + self.npml)
            self.rec_top_right_const = np.copy(a[0])
            self.rec_top_right_var = np.copy(a[1])

            a = np.int32(rec_loc[self.n_well_rec, :]/ self.dh + self.npml)
            self.rec_surface_const = np.copy(a[1])
            self.rec_surface_var = np.copy(a[0])

            self.rec_top_left_const = np.int32(rec_loc[0, 0] / self.dh + self.npml)
            self.rec_top_left_var = np.int32(rec_loc[:, 1] / self.dh + self.npml)[0]
            self.src_cts = src.j[0]

        # ======== Parameters Boundary condition ======
        self.dx_pml, self.dz_pml = tools.pml_counstruction(self.tnz, self.tnx, self.dh, self.npml,
                                                     self.pmlR, self.pml_dir)

        self.components = components

        # TODO: Check sources
        # if self.acq_type == 0:
        #     self.injSrc = self.injSrc
        # elif self.acq_type == 1:
        #     self.injSrc = self.injSrc


        v = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        # Buffer for forward modelling
        self.initiate_wave()

        # TODO: Can be just for pressure
        # Make a list for seismograms
        self.seismogram = {
            'vx': torch.zeros((self.nt, self.nr, self.ns)),
            'vz': torch.zeros((self.nt, self.nr, self.ns)),
            'taux': torch.zeros((self.nt, self.nr, self.ns)),
            'tauz': torch.zeros((self.nt, self.nr, self.ns)),
            'tauxz': torch.zeros((self.nt, self.nr, self.ns))
        }
        
    def initiate_wave(self):
        # Buffer for forward modelling
        self.vx_b = torch.zeros((self.tnz, self.tnx))
        self.vz_b = torch.zeros((self.tnz, self.tnx))
        self.taux_b = torch.zeros((self.tnz, self.tnx))
        self.tauz_b = torch.zeros((self.tnz, self.tnx))
        self.tauxz_b = torch.zeros((self.tnz, self.tnx))

    def elastic_update_v(self):
        self.vx_b += - self.dt * self.vdx_pml_b * self.vx_b + \
            self.dt * (1/self.rho_b) * (self.dxm(self.taux_b, self.dh) + self.dzm(self.tauxz_b, self.dh))
        self.vz_b += - self.dt * self.vdz_pml_b * self.vz_b + \
            self.dt * (1/self.rho_b) * (self.dxp(self.tauxz_b, self.dh) + self.dzp(self.tauz_b, self.dh))

    def elastic_update_tau(self):
        lmu = (self.lam_b + 2 * self.mu_b)
            
        self.taux_b += - self.dt * self.vdx_pml_b * self.taux_b + \
                self.dt * lmu * self.dxp(self.vx_b, self.dh) + self.dt * self.lam_b * self.dzm(self.vz_b, self.dh)

        self.tauz_b += - self.dt * self.vdz_pml_b * self.tauz_b + \
                self.dt * self.lam_b * self.dxp(self.vx_b, self.dh) + self.dt * lmu * self.dzm(self.vz_b, self.dh)

        self.tauxz_b += - self.dt * (self.vdx_pml_b + self.vdz_pml_b) * self.tauxz_b + \
                self.dt * self.mu_b * (self.dxm(self.vz_b, self.dh) + self.dzp(self.vx_b, self.dh))


    def make_seismogram(self, s, t):
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
        b = self.taux_b.clone().detach()
        def get_from_gpu(buffer):
            # a = buffer.clone().detach()

            return buffer[rec_loc[:, 1], rec_loc[:, 0]]

        self.seismogram['vx'][np.int32(t - 1), :, s] = \
            get_from_gpu(self.vx_b)

        self.seismogram['vz'][np.int32(t - 1), :, s] = \
            get_from_gpu(self.vz_b)

        self.seismogram['taux'][np.int32(t - 1), :, s] = \
            get_from_gpu(self.taux_b)

        self.seismogram['tauz'][np.int32(t - 1), :, s] = \
            get_from_gpu(self.tauz_b)

        self.seismogram['tauxz'][np.int32(t - 1), :, s] = \
            get_from_gpu(self.tauxz_b)

    def pml_preparation(self, v_max):

        self.vdx_pml_b = torch.tensor(self.dx_pml) * v_max
        self.vdz_pml_b = torch.tensor(self.dz_pml) * v_max

   
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
        
        self.mu_b = model['rho'] * (model['vs'] ** 2)

        self.lam_b = model['rho'] * (model['vp'] ** 2) - 2 * self.mu_b

        self.rho_b = model['rho']



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
    def __init__(self, inpa, src, rec_loc, model_shape, n_well_rec=None, chpr=10, components=0, set_env_variable=True):
        WavePreparation.__init__(self, inpa, src, rec_loc, model_shape, n_well_rec, chpr=chpr,
                                 components=components, set_env_variable=set_env_variable)
        
        self.device = 'cpu'  # TODO: Change
        Fdm.__init__(self, 2 * self.sdo, tensor='torch', device=self.device)
        
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
            self.initiate_wave()

            self.__kernel(s)

        if self.acq_type == 2:
            for par in self.seismogram:
                self.seismogram[par][:, :self.n_well_rec] = np.flip(self.seismogram[par][:, :self.n_well_rec], axis=1)
        return self.seismogram

    def __kernel(self, s):
        showpurose = np.zeros((self.tnz, self.tnx), dtype=np.float32)

        for t in range(self.nt):
            src_v_x, src_v_z, src_t_x, src_t_z, src_t_xz = np.float32(self.src(t))

            inj_src(self.vx_b, self.vz_b, self.taux_b, self.tauz_b,self.tauxz_b,
                   self.srcx[s], self.srcz[s],
                   src_v_x, src_v_z,src_t_x, src_t_z,src_t_xz)

            self.elastic_update_v()

            self.elastic_update_tau()
            
            self.make_seismogram(s, t)

            if self.forward_show and np.remainder(t, 20) == 0:
                showpurose = np.copy(self.taux_b.detach().numpy())
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
        data = acq.seismic_section(seismo, self.components, shape='3d')
        return data


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


def inj_src(vx_b, vz_b, taux_b, tauz_b,tauxz_b,
           srcx, srcz,
           src_v_x, src_v_z, src_t_x, src_t_z,src_t_xz):
        
    vx_b[srcz, srcx] += torch.tensor(src_v_x)
    vz_b[srcz, srcx] += torch.tensor(src_v_z)
    taux_b[srcz, srcx] += torch.tensor(src_t_x)
    tauz_b[srcz, srcx] += torch.tensor(src_t_z)
    tauxz_b[srcz, srcx] += torch.tensor(src_t_xz)




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
