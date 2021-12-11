import numpy as np
import PyFWI.processing as seis_process
import PyFWI.fwi_tools as tools
from PyFWI.fwi_tools import recorder, expand_model, CPML
import pyopencl as cl
import os
from pyopencl.tools import get_test_platforms_and_devices
import matplotlib.pyplot as plt
import PyFWI.seiplot as seiplt

class wave_preparation():
    def __init__(self, inpa, src, rec_loc, model_size, n_well_rec=0, chpr=10, components=4):
        '''
        A class to prepare the variable and basic functions for wave propagation.
        
        '''
        #TODO: work on how ypu specify the acq_type, getting n_well_rec, using that again fpr two .cl files
        keys = [*inpa]
        
        self.t = inpa['t']
        self.dt = inpa['dt']
        self.nt = int(self.t // self.dt)
        
        self.nx = np.int32(model_size[1])
        self.nz = np.int32(model_size[0])
        
        if 'npml' in keys:
            self.npml = inpa['npml']
        else:
            self.npml = 0
            
        # Number of samples in x- and z- direction by considering pml
        self.tnx = np.int32(self.nx + 2 * self.npml)
        self.tnz = np.int32(self.nz + 2 * self.npml)
        
        self.dh = inpa['dh']
        
        if 'sdo' in keys:
            self.sdo = inpa['sdo']
        else:
            self.sdo = 4
            
        self.srcx = np.int32(src.i + inpa['npml'])
        self.srcz = np.int32(src.j + inpa['npml'])
        
        self.src = src
        self.ns = np.int32(src.i.size)
        
        self.dxr = np.int32(inpa['rec_dis'] / self.dh)
        
        self.chpr = chpr
        chp = int(chpr * self.nt / 100)
        self.chp = np.linspace(1, self.nt-1, chp, dtype=np.int32)
        if (chpr != 0) & (len(self.chp) < 2):
            self.chp = np.array([1, self.nt])
            
        self.nchp = len(self.chp)
        # Take chpr into account 
        
        self.rec_loc = rec_loc
        self.nr = rec_loc.shape[0]
        self.n_well_rec = n_well_rec
        
        self.acq_type = inpa["acq_type"]
        if n_well_rec ==0 and self.acq_type == 2:
            raise Exception(" Number of geophons in the wells is not defined")
        
        # The matrix containg the geometry of acquisittion (Never used really)
        data_guide = np.zeros((6, self.nr * self.ns))
        data_guide[0, :] = np.kron(np.arange(self.ns), np.ones(self.nr))
        data_guide[1, :] = np.kron(self.srcx * self.dh, np.ones(self.nr))
        data_guide[2, :] = np.kron(self.srcz * self.dh, np.ones(self.nr))
        data_guide[3, :] = np.kron(np.ones(self.ns), rec_loc[:, 0])
        data_guide[4, :] = np.kron(np.ones(self.ns), rec_loc[:, 1])
        if inpa["acq_type"] == 2:
            data_guide[4, :int(self.n_well_rec)] = np.flip(data_guide[4, :int(self.n_well_rec)])
        data_guide[5, :] = np.abs(data_guide[1, :] - data_guide[3, :])
        
        self.data_guide_sampling = np.copy(data_guide).astype(np.int32)
        self.data_guide_sampling[1:, :] = np.int32(self.data_guide_sampling[1:, :] / self.dh)
        self.data_guide_sampling[1:5, :] += inpa['npml'] 
        
        if inpa["acq_type"] == 0:
            self.rec_cts = np.int32(rec_loc[0, 0] / self.dh + inpa['npml'])
            self.rec_var = np.int32(rec_loc[:, 1] / self.dh + inpa['npml'])
            self.src_cts = src.i[0]
            
        elif inpa["acq_type"] in [1, 2]:
            self.rec_cts = np.int32(rec_loc[0, 1] / self.dh + inpa['npml'])
            self.rec_var = np.int32(rec_loc[:, 0] / self.dh + inpa['npml'])
            self.src_cts = src.j[0]

        # ======== Parameters Boundary condition ======
        self.dx_pml, self.dz_pml = tools.pml_counstruction(self.tnz, self.tnx, self.dh, self.npml,
                                                     inpa['pmlR'], inpa['pml_dir'])

        if 'energy_balancing' in keys:
            self.energy_balancing = inpa['energy_balancing']
        else:
            self.energy_balancing = False
            
        self.W = {
            'vx': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32), 
            'vz': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'taux': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'tauz': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'tauxz': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
        }
        
        self.Lam = {
            'avx': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32), 
            'avz': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'ataux': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'atauz': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
            'atauxz': np.zeros((self.tnz, self.tnx, self.ns, self.nchp), dtype=np.float32),
        }

        self.D = seis_process.derivatives(order=self.sdo)
        
        self.components = components
        self.R = recorder(self.nt, self.rec_loc, self.ns, self.dh)
        
        # To call openCl
        # Select the platform (if not provided, pick 0)
        if "platform" in keys:
            platform = inpa["platform"]
        else:
            platform = 0
            
        # Choose th device (pick 0 if not provided)
        devices = get_test_platforms_and_devices()[0][1]
        if "device" in keys:
            device = inpa["device"]
            if device >= len(devices):
                raise Exception("Bad chosen device. There are {} available device(s).".format(len(devices)))
        else:
            device = 0
            print("Device {} is chosen.".format(device))

        os.environ['PYOPENCL_CTX'] = str(platform) + ':' + str(device)
        os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        
        self.prg = cl.Program(self.ctx, self.kernel_caller()).build()
        self.mf = cl.mem_flags
        
        # Buffer for residuals
        # res = np.zeros((np.int32(self.nr))).astype(np.float32, order='C')
        res = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        self.res_vx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                  self.mf.COPY_HOST_PTR, hostbuf=res)
        self.res_vz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                  self.mf.COPY_HOST_PTR, hostbuf=res)
        self.res_taux_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                    self.mf.COPY_HOST_PTR, hostbuf=res)
        self.res_tauz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                    self.mf.COPY_HOST_PTR, hostbuf=res)
        self.res_tauxz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                     self.mf.COPY_HOST_PTR, hostbuf=res)
                    
                    
        v = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        
        # Buffer for forward modelling
        self.vx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.vz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.taux_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                self.mf.COPY_HOST_PTR, hostbuf=v)
        self.tauz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                self.mf.COPY_HOST_PTR, hostbuf=v)
        self.tauxz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                 self.mf.COPY_HOST_PTR, hostbuf=v)
                
        self.glam_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.gmu_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.grho_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        
        # Buffer for seismograms
        seismogram_id = np.zeros((1, self.nr)).astype(np.float32, order='C')

        self.seismogramid_vx_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)
        self.seismogramid_vz_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)
        self.seismogramid_taux_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)
        self.seismogramid_tauz_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)
        self.seismogramid_tauxz_b = cl.Buffer(
            self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=seismogram_id)

        # Make a list for seismograms
        self.seismogram = {
            'vx': np.zeros((self.nt, self.nr * self.ns)).astype(np.float32, order='C'),
            'vz': np.zeros((self.nt, self.nr * self.ns)).astype(np.float32, order='C'),
            'taux': np.zeros((self.nt, self.nr * self.ns)).astype(np.float32, order='C'),
            'tauz': np.zeros((self.nt, self.nr * self.ns)).astype(np.float32, order='C'),
            'tauxz': np.zeros((self.nt, self.nr * self.ns)).astype(np.float32, order='C'),
        }
    
    def adjoint_buffer_preparing(self):
        # Buffer for gradient
        g_mu = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.Gmu_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_mu)

        g_lam = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.Glam_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_lam)

        g_rho = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.Grho_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_rho)

        g_mu_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.g_mu_precond_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_mu_precond)

        self.g_lam_precond_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_mu_precond)

        self.g_rho_precond_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=g_mu_precond)

        # Buffer for backward modelling
        v = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        self.avx_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.avz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                              self.mf.COPY_HOST_PTR, hostbuf=v)
        self.ataux_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                self.mf.COPY_HOST_PTR, hostbuf=v)
        self.atauz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                self.mf.COPY_HOST_PTR, hostbuf=v)
        self.atauxz_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                 self.mf.COPY_HOST_PTR, hostbuf=v)
        
    def gradient_reading(self):
        g_mu = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        g_lam = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

        g_rho = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
        
        cl.enqueue_copy(self.queue, g_mu, self.Gmu_b)
        cl.enqueue_copy(self.queue, g_lam, self.Glam_b)
        cl.enqueue_copy(self.queue, g_rho, self.Grho_b)
        
        if self.energy_balancing:
            g_mu_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_lam_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_rho_precond = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')

            cl.enqueue_copy(self.queue, g_mu_precond, self.g_mu_precond_b)
            cl.enqueue_copy(self.queue, g_lam_precond, self.g_lam_precond_b)
            cl.enqueue_copy(self.queue, g_rho_precond, self.g_rho_precond_b)
            print("balanced")
        else:
            g_mu_precond = np.ones((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_lam_precond = np.ones((self.tnz, self.tnx)).astype(np.float32, order='C')
            g_rho_precond = np.ones((self.tnz, self.tnx)).astype(np.float32, order='C')
            print("Not balanced")

        def denom2factor(precond0):
            precond = np.copy(precond0)

            factor = np.zeros((self.nz, self.nx), np.float32)

            denom = np.sqrt(precond)
            denom = denom[self.npml + sdo:self.tnz - self.npml - sdo, self.npml + sdo:self.tnx - self.npml - sdo]
            factor[sdo:-sdo, sdo:-sdo] = 1 / denom
            factor = factor / np.abs(factor).max()

            return np.copy(factor)

        sdo = self.sdo

        factor_rho = denom2factor(g_rho_precond)
        grho = g_rho[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml] * factor_rho

        factor_mu = denom2factor(g_mu_precond)
        gmu = g_mu[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml] * factor_mu

        factor_lam = denom2factor(g_lam_precond)
        glam = g_lam[self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml] * factor_lam

        return glam, gmu, grho
        

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

        # Make space to read data of each time step from the buffer
        def get_from_opencl(buffer):
            # seismogram_id = np.zeros((1, self.nr)).astype(np.float32, order='C')
            # cl.enqueue_copy(self.queue, seismogram_id, buffer)
            
            seismogram_component = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
            cl.enqueue_copy(self.queue, seismogram_component, buffer)
            return seismogram_component
            # return np.copy(seismogram_id)

        # Getting data from opencl
        # cl.enqueue_copy(self.queue, seismogram_id, self.seismogramid_vx_b)
        
        vx = get_from_opencl(self.vx_b)
        cond = self.data_guide_sampling[0, :] == s
        data_guide_sampling = self.data_guide_sampling[:, cond]
        
        vx = get_from_opencl(self.vx_b)
        self.seismogram['vx'][np.int32(t), s * self.nr:(s + 1) * self.nr] = \
            np.copy(vx[data_guide_sampling[4, :], data_guide_sampling[3, :]])
            # get_from_opencl(self.seismogramid_vx_b)

        vx = get_from_opencl(self.vz_b)
        self.seismogram['vz'][np.int32(t), s * self.nr:(s + 1) * self.nr] = \
            np.copy(vx[data_guide_sampling[4, :], data_guide_sampling[3, :]])
            # get_from_opencl(self.seismogramid_vz_b)

        vx = get_from_opencl(self.taux_b)
        self.seismogram['taux'][np.int32(t), s * self.nr:(s + 1) * self.nr] = \
            np.copy(vx[data_guide_sampling[4, :], data_guide_sampling[3, :]])
            # get_from_opencl(self.seismogramid_taux_b)

        vx = get_from_opencl(self.tauz_b)
        self.seismogram['tauz'][np.int32(t), s * self.nr:(s + 1) * self.nr] = \
            np.copy(vx[data_guide_sampling[4, :], data_guide_sampling[3, :]])
            # get_from_opencl(self.seismogramid_tauz_b)

        vx = get_from_opencl(self.tauxz_b)
        self.seismogram['tauxz'][np.int32(t), s * self.nr:(s + 1) * self.nr] = \
            np.copy(vx[data_guide_sampling[4, :], data_guide_sampling[3, :]])
            # get_from_opencl(self.seismogramid_tauxz_b)

    def make_residual(self, res, s, t):
        """
        This function reads the inject the residual to residual buffer based on source and the
        time step.

        Parameters
        ----------
            res: list
                list containing the residual of all parameters

            s : int
                Number of current acive source.

            t : float
                Current time step.
        """
        # Injection data into opencl
        def prepare_to_opencl(residul):
            cond = self.data_guide_sampling[0, :] == s
            data_guide_sampling = self.data_guide_sampling[:, cond]
            
            res_vx = np.zeros((self.tnz, self.tnx)).astype(np.float32, order='C')
            res_vx[data_guide_sampling[4, :], data_guide_sampling[3, :]] = residul[np.int32(t), s * self.nr:(s + 1) * self.nr]
            return res_vx
        
        res_vx = prepare_to_opencl(res['vx'])
        # res_src_vx = (res['vx'][np.int32(t), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        cl.enqueue_copy(self.queue, self.res_vx_b, res_vx)

        res_vz = prepare_to_opencl(res['vx'])
        res_src_vz = (res['vz'][np.int32(t), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        cl.enqueue_copy(self.queue, self.res_vz_b, res_vz)
        
        res_taux = prepare_to_opencl(res['taux'])
        res_src_taux = (res['taux'][np.int32(t), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        cl.enqueue_copy(self.queue, self.res_taux_b, res_taux)
        
        res_tauz = prepare_to_opencl(res['taux'])
        res_src_tauz = (res['tauz'][np.int32(t), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        cl.enqueue_copy(self.queue, self.res_tauz_b, res_tauz)

        res_tauxz = prepare_to_opencl(res['tauxz'])
        res_src_tauxz = (res['tauxz'][np.int32(t), s * self.nr:(s + 1) * self.nr]).astype(np.float32, order='C')
        cl.enqueue_copy(self.queue, self.res_tauxz_b, res_tauxz)


    def pml_preparation(self, v_max):
        
        vdx_pml = self.dx_pml * v_max
        vdz_pml = self.dz_pml * v_max
        self.vdx_pml_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=vdx_pml)
        self.vdz_pml_b = cl.Buffer(self.ctx, self.mf.READ_WRITE |
                                   self.mf.COPY_HOST_PTR, hostbuf=vdz_pml)


    def kernel_caller(self):
        '''
            This function is used to specify the constants for use in openCl's
            files and choose the asked openCl source to run based on the assumptions
            of problem.
            
            The coeeficients are based on Lavendar, 1988 and Hasym et al., 2014.

            Returns
            -------

                kernel_source : str
                    macro plus the openCl's source.
        '''
        if self.sdo == 4:  # For O4
            c1 = 9 / 8
            c2 = -1 / 24
            c3 = 0
            c4 = 0
        elif self.sdo == 8:  # For O8
            c1 = 1715 / 1434  # 1.2257
            c2 = -114 / 1434  # - 0.099537
            c3 = 14 / 1434  # 0.018063
            c4 = -1 / 1434  # - 0.0026274

        macro = """
            #define Nz	   %d
            #define Nx     %d
            #define Ns     %d
            #define dt     %f
            #define dx     %f
            #define dz     %f
            #define npml   %d

            #define sdo     %d
            #define c1     %f
            #define c2     %f
            #define c3     %f
            #define c4     %f

            #define center	   (i)*Nx + (j) 

            #define below    (i+1)*Nx + (j)
            #define below2   (i+2)*Nx + (j)
            #define below3   (i+3)*Nx + (j)
            #define below4   (i+4)*Nx + (j)

            #define above     (i-1)*Nx + (j)
            #define above2    (i-2)*Nx + (j)
            #define above3    (i-3)*Nx + (j)
            #define above4    (i-4)*Nx + (j)

            #define right     (i)*Nx + (j+1)
            #define right2    (i)*Nx + (j+2)
            #define right3    (i)*Nx + (j+3)
            #define right4    (i)*Nx + (j+4)

            #define left    (i)*Nx + (j-1)
            #define left2   (i)*Nx + (j-2)
            #define left3   (i)*Nx + (j-3)
            #define left4   (i)*Nx + (j-4)
            
            #define src_depth %d

                //float Diff_Forward(float , float , float , float , int );
                //float Diff_Backward(float , float , float , float , int );
        """ % (self.tnz, self.tnx, self.ns, self.dt, self.dh, self.dh, self.npml,
               self.sdo,
               c1, c2, c3, c4, self.src_cts)

        # Decide on openCl file based on medium type
        cl_file_name = "elastic_velocity.cl"

        print(cl_file_name)
        # Call the openCl file
        path = os.path.dirname(__file__) + "/"
        f = open(path + cl_file_name, 'r')
        fstr = "".join(f.readlines())
        kernel_source = macro + fstr

        return kernel_source
    
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
            self.__im0 = ax.imshow(model[key][self.npml:self.tnz - self.npml, self.npml:self.tnx - self.npml], cmap='jet',
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
        
        self.mu = model['rho'] * (model['vs'] ** 2)
        
        self.lam = model['rho'] * (model['vp'] ** 2) - 2 * self.mu
        
        self.rho = model['rho']
        
        self.vp = model['vp']
        
        self.vs = model['vs']
        
        self.rho_b = cl.Buffer(self.ctx, self.mf.COPY_HOST_PTR,
                               hostbuf=1 / model['rho'])

        self.mu_b = cl.Buffer(self.ctx, self.mf.COPY_HOST_PTR,
                              hostbuf=self.mu)
        self.lam_b = cl.Buffer(self.ctx, self.mf.COPY_HOST_PTR,
                               hostbuf=self.lam)
        
        
class wave_propagator(wave_preparation):
    def __init__(self, inpa, src, rec_loc, model_size, n_well_rec=None, chpr=10, components=4):
        wave_preparation.__init__(self, inpa, src, rec_loc, model_size, n_well_rec, chpr=chpr, components=components)
        # CPML.__init__(self, self.dh, self.dt, N=self.npml, nd=2, Rc=1e-5, nu0=1, nnu=2, nalpha=1, alpha0=0)
    
    def forward_propagator(self, model):
        """ This function is in charge of forward modelling for acoustic case

        Parameters
        ---------
            model: dictionary
                A dictionary containing p-wave velocity and density

        """
        coeff = np.int32(+1)

        if self.forward_show:
            self.initial_wavefield_plot(model)

        for s in range(self.ns):
            self.prg.MakeAllZero(self.queue, (self.tnz, self.tnx), None,
                                 self.vx_b, self.vz_b,
                                 self.taux_b, self.tauz_b, self.tauxz_b)
            
            self.__kernel(s, coeff) 
            
        return self.seismogram
            
    def __kernel(self, s, coeff=+1):

        chpc = 0
        showpurose = np.zeros((self.tnz, self.tnx), dtype=np.float32)

        for t in range(1, self.nt):
            src_kt_x, src_kt_z = np.float32(self.src(t))

            self.prg.injSrc(self.queue, (self.tnz, self.tnx), None,
                            self.vx_b, self.vz_b,
                            self.taux_b, self.tauz_b, self.tauxz_b,
                            self.rho_b,
                            self.seismogramid_vx_b, self.seismogramid_vz_b,
                            self.seismogramid_taux_b, self.seismogramid_tauz_b, self.seismogramid_tauxz_b,
                            # self.dxr, self.rec_cts, self.rec_var,
                            self.srcx[s], self.srcz[s],
                            src_kt_x, src_kt_z)

            self.prg.update_velx(self.queue, (self.tnz, self.tnx), None,
                                 coeff,
                                 self.vx_b,
                                 self.taux_b, self.tauxz_b,
                                 self.rho_b,
                                 self.vdx_pml_b, self.vdz_pml_b
                                 )

            self.prg.update_velz(self.queue, (self.tnz, self.tnx), None,
                                 coeff,
                                 self.vz_b,
                                 self.tauz_b, self.tauxz_b,
                                 self.rho_b,
                                 self.vdx_pml_b, self.vdz_pml_b
                                 )

            self.prg.update_tauz(self.queue, (self.tnz, self.tnx), None,
                                 coeff,
                                 self.vx_b, self.vz_b,
                                 self.taux_b, self.tauz_b,
                                 self.lam_b, self.mu_b,
                                 self.vdx_pml_b, self.vdz_pml_b
                                 )

            self.prg.update_tauxz(self.queue, (self.tnz, self.tnx), None,
                                  coeff,
                                  self.vx_b, self.vz_b,
                                  self.tauxz_b,
                                  self.mu_b,
                                  self.vdx_pml_b, self.vdz_pml_b
                                  )

            copy_purpose = np.zeros((self.tnz, self.tnx), dtype=np.float32)
            if t == self.chp[chpc]:
                cl.enqueue_copy(self.queue, copy_purpose, self.vx_b)
                self.W['vx'][:, :, s, chpc] = np.copy(copy_purpose)

                cl.enqueue_copy(self.queue, copy_purpose, self.vz_b)
                self.W['vz'][:, :, s, chpc] = np.copy(copy_purpose)

                cl.enqueue_copy(self.queue, copy_purpose, self.taux_b)
                self.W['taux'][:, :, s, chpc] = np.copy(copy_purpose)

                cl.enqueue_copy(self.queue, copy_purpose, self.tauz_b)
                self.W['tauz'][:, :, s, chpc] = np.copy(copy_purpose)

                cl.enqueue_copy(self.queue, copy_purpose, self.tauxz_b)
                self.W['tauxz'][:, :, s, chpc] = np.copy(copy_purpose)

                chpc += 1

            # cl.enqueue_copy(self.queue, self.seismogram_id, self.seismogramid_b)
            # print(self.seismogram_id.max())
            self.make_seismogram(s, t)

            if self.forward_show and np.remainder(t, 20) == 0:
                cl.enqueue_copy(self.queue, showpurose, self.tauz_b)
                self.plot_propagation(showpurose, t)
        # return 
    
    def __adjoint_modelling_per_source(self, res):
        self.prg.MakeGradZero(self.queue, (self.tnz, self.tnx), None,
                              self.Gmu_b, self.Glam_b, self.Grho_b,
                              self.g_mu_precond_b, self.g_lam_precond_b, self.g_rho_precond_b)

        for s in range(self.ns):
            self.prg.MakeAllZero(self.queue, (self.tnz, self.tnx), None,
                                 self.vx_b, self.vz_b,
                                 self.taux_b, self.tauz_b, self.tauxz_b)
            
            self.prg.MakeAllZero(self.queue, (self.tnz, self.tnx), None,
                                 self.avx_b, self.avz_b,
                                 self.ataux_b, self.atauz_b, self.atauxz_b)

            self.__kernel_gradient(res, s)
            
    def __kernel_gradient(self, res, s, coeff=-1):
        chpc = self.nchp - 1

        coeff = np.int32(coeff)

        vx = np.zeros((self.tnz, self.tnx), dtype=np.float32)
        adj_vx = np.zeros((self.tnz, self.tnx), dtype=np.float32)

        # time loop
        for t in range(self.nt - 1, 0, -1):  # range(self.nt-1,self.nt-2,-1):#
            self.make_residual(res, s, t)

            if t == self.chp[chpc]:
                self.vx = np.copy(self.W['vx'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.vx_b, self.vx)

                self.vz = np.copy(self.W['vz'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.vz_b, self.vz)

                self.taux = np.copy(self.W['taux'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.taux_b, self.taux)

                self.tauz = np.copy(self.W['tauz'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.tauz_b, self.tauz)

                self.tauxz = np.copy(self.W['tauxz'][:, :, s, chpc])
                cl.enqueue_copy(self.queue, self.tauz_b, self.tauxz)

                chpc -= 1
 
            else:
                """ Backward propagation  """
                self.prg.update_tauxz(self.queue, (self.tnz, self.tnx), None,
                                      coeff,
                                      self.vx_b, self.vz_b,
                                      self.tauxz_b,
                                      self.mu_b,
                                      self.vdx_pml_b, self.vdz_pml_b
                                      )

                self.prg.update_tauz(self.queue, (self.tnz, self.tnx), None,
                                     coeff,
                                     self.vx_b, self.vz_b,
                                     self.taux_b, self.tauz_b,
                                     self.lam_b, self.mu_b,
                                     self.vdx_pml_b, self.vdz_pml_b
                                     )

                self.prg.update_velz(self.queue, (self.tnz, self.tnx), None,
                                     coeff,
                                     self.vz_b,
                                     self.tauz_b, self.tauxz_b,
                                     self.rho_b,
                                     self.vdx_pml_b, self.vdz_pml_b
                                     )

                self.prg.update_velx(self.queue, (self.tnz, self.tnx), None,
                                     coeff,
                                     self.vx_b,
                                     self.taux_b, self.tauxz_b,
                                     self.rho_b,
                                     self.vdx_pml_b, self.vdz_pml_b
                                     )

            """ Adjoint modeling """
            self.prg.Adj_injSrc(self.queue, (self.tnz, self.tnx), None,
                                self.avx_b, self.avz_b,
                                self.ataux_b, self.atauz_b, self.atauxz_b,
                                self.res_vx_b, self.res_vz_b,
                                self.res_taux_b, self.res_tauz_b, self.res_tauxz_b,
                                self.dxr, self.rec_cts,
                                self.rec_var, self.rec_var)

            self.prg.Adj_update_tau(self.queue, (self.tnz, self.tnx), None,
                                    self.avx_b, self.avz_b,
                                    self.ataux_b, self.atauz_b, self.atauxz_b,
                                    self.rho_b,
                                    self.vdx_pml_b, self.vdz_pml_b)

            self.prg.Adj_update_v(self.queue, (self.tnz, self.tnx), None,
                                  self.avx_b, self.avz_b,
                                  self.ataux_b, self.atauz_b, self.atauxz_b,
                                  self.lam_b, self.mu_b,
                                  self.vdx_pml_b, self.vdz_pml_b
                                  )

            self.prg.Grad_mu(self.queue, (self.tnz, self.tnx), None,
                             self.vx_b, self.vz_b,
                             self.taux_b, self.tauz_b, self.tauxz_b,
                             self.ataux_b, self.atauz_b, self.atauxz_b,
                             self.Gmu_b, self.g_mu_precond_b)

            self.prg.Grad_lam(self.queue, (self.tnz, self.tnx), None,
                              self.vx_b, self.vz_b,
                              self.taux_b, self.tauz_b,
                              self.ataux_b, self.atauz_b,
                              self.Glam_b, self.g_lam_precond_b)

            self.prg.Grad_rho(self.queue, (self.tnz, self.tnx), None,
                              self.vx_b, self.vz_b,
                              self.taux_b, self.tauz_b, self.tauxz_b,
                              self.avx_b, self.avz_b,
                              self.rho_b, self.Grho_b, self.g_rho_precond_b)

            # Plotting wave propagation
            if self.backward_show and (np.remainder(t, 20) == 0 or t == self.nt - 2):
                cl.enqueue_copy(self.queue, vx, self.tauxz_b)

                cl.enqueue_copy(self.queue, adj_vx, self.avx_b)

                self.plot_propagation(vx, t, adj_vx)
                
            
    def forward_modeling(self, model0, show=False):
        self.forward_show = show
        model = model0.copy()

        for params in model:
            # model[params] = model[params]  # To avoid sticking BC. to the original model
            model[params] = expand_model(model[params], self.tnz, self.tnx, self.npml)
        
        self.pml_preparation(model['vp'].max())
        self.elastic_buffers(model)
        seismo = self.forward_propagator(model)    
        return seismo
    
    def gradient(self, res, show):
        self.backward_show = show
        self.adjoint_buffer_preparing()
        
        if show:
            self.initial_wavefield_plot({'vp':self.vp}, plot_type="Backward")
        
        self.__adjoint_modelling_per_source(res)
        
        glam, gmu, grho0 = self.gradient_reading()

        gvp, gvs, grho = tools.grad_lmd_to_vd(glam, gmu, grho0,
                                              self.lam[self.npml: self.tnz-self.npml, self.npml: self.tnx-self.npml],
                                              self.mu[self.npml: self.tnz-self.npml, self.npml: self.tnx-self.npml],
                                              self.rho[self.npml: self.tnz-self.npml, self.npml: self.tnx-self.npml])
        
        return {'vp': gvp,
                'vs': gvs,
                'rho': grho
                }
        
        
if __name__ == "__main__":
    import PyFWI.model_dataset as md
    import PyFWI.acquisition as acq

    model_gen = md.ModelGenerator('yang') # louboutin') # 
    
    model = model_gen()
    # model_gen.show(['vs'])
    model_shape = model[[*model][0]].shape
    
    inpa = {}
    # Number of pml layers
    inpa['npml'] = 20
    inpa['pmlR'] = 1e-5
    inpa['pml_dir'] = 2
    inpa['device'] = 2
    inpa['energy_balancing'] = False
    chpr = 1
    sdo = 4
    fdom = 25
    fn = 125
    vp = model['vp']
    D = seis_process.derivatives(order=sdo)
    dh = vp.min()/(D.dh_n * fn)
    dh = 2.
    inpa['dh'] = dh
    
    dt = D.dt_computation(vp.max(), inpa['dh'])
    inpa['dt'] = dt
    
    print(f'{dh = } ........... {dt = }')
    inpa['t'] = 0.36
    
    offsetx = inpa['dh'] * model_shape[1]
    depth = inpa['dh'] * model_shape[0]

    inpa['rec_dis'] = inpa['dh']
    ns = 5
    inpa['acq_type'] = 0

    src_loc, rec_loc, n_surface_rec, n_well_rec = acq.AcqParameters(ns, inpa['rec_dis'], offsetx, depth, inpa['dh'], sdo, inpa['acq_type'])
    
    src = acq.Source(src_loc, inpa['dh'], inpa['dt'])
    src.Ricker(fdom)
    
    W = wave_propagator(inpa, src, rec_loc, model_shape, n_well_rec, chpr=chpr)
    d_obs = W.forward_modeling(model, False)
    
    m0 = model_gen(vintage=1, smoothing=True)
    
    Lam = wave_propagator(inpa, src, rec_loc, model_shape, n_well_rec)
    d_est = Lam.forward_modeling(m0, show=False)

    res = tools.residual(d_est, d_obs)
        
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 3, 1)
    # seiplt.seismic_section(ax, d_obs['taux'], vmin=d_obs['taux'].min() / 5, vmax=d_obs['taux'].max() / 5)
    # ax = fig.add_subplot(1, 3, 2)
    # seiplt.seismic_section(ax, d_est['taux'])
    # ax = fig.add_subplot(1, 3, 3)
    # seiplt.seismic_section(ax, res['taux'])
    # plt.show()
    
    grad = Lam.gradient(res, show=False)
    
    seiplt.earth_model(grad)
    plt.show()
    a=1