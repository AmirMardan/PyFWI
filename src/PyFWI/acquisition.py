from logging import exception
import numpy as np


def acq_parameters(ns, rec_dis, offsetx, depth, dh, sdo, acq_type):
    """
    A function to define the acquisition based on user's demand

    Parameters
    ----------
        INPA : dictionary
            A dictionnary containing required parameters for iversion, at least:
                - ns: Number of sources
                - rec_dis: Distance between receivers
                - offsetx: Length of acquisition in x-direction
                - depth: Depth of acquisition
                - dh: spatial sampling rate
                - sdo: Spatial differentiation order
                - acq_type: Type of acquisition (0: crosswell, 1: surface, 2: both)

    Returns
    --------
        src_loc: float32
            Location of sources
        rec_loc: float32
            Location of receivers
        n_surface_rec: int
            Number of receivers at the surface
        n_well_rec: int
            Number of receivers in wells at each side of the model
    """
    if rec_dis < dh:
        raise Exception("Receiver distance should be larger than spatial sampling")
    if acq_type == 0:
        # In case of crosswell seismic

        src_loc, rec_loc = crosswell(ns, rec_dis, offsetx, depth,
                         dh, sdo)
        n_surface_rec = 0
        n_well_rec = rec_loc.shape[0]
    elif acq_type == 1:
        src_loc, rec_loc = surface_seismic(ns, rec_dis, offsetx, dh, sdo)

        n_surface_rec = rec_loc.shape[0]
        n_well_rec = 0
    elif acq_type == 2:
        src_loc, rec_loc = surface_seismic(ns, rec_dis, offsetx, dh, sdo)
        n_surface_rec = rec_loc.shape[0]

        _, rec_loc2 = crosswell(ns, rec_dis, offsetx, depth,
                                     dh, sdo)

        rec_loc1_cst = np.ones((rec_loc2.shape), np.float32) @\
                   (np.array([[rec_loc[0, 1]], [0]], np.float32))
        rec_loc1 = np.hstack((rec_loc1_cst, rec_loc2[:, 1].reshape(-1,1)))
        rec_loc = np.vstack((rec_loc1, rec_loc, rec_loc2))
        n_well_rec = rec_loc2.shape[0]

    return src_loc, rec_loc, n_surface_rec, n_well_rec


def crosswell(ns, rec_dis, offsetx, depth,
              dh, sdo):
    """
    A function to design a crosswell acquisition

    Parameters
    ----------
        ns : int
            Number of sources

        rec_dis : float32
            Distance between receivers

        offsetx : flloat32
            Length of survey in x-direction

        depth : float32
            Depth of survey

        dh : float32
            Sampling rate

        sdo: {2, 4, 8}
            Spatial order of finite difference method

    Returns
    -------

        src_loc: float32
            Location of sources
        rec_loc: float32
            Location of receivers
    """
    # Distribution of receivers
    rec_locZ = np.arange(2*rec_dis, depth-(2*rec_dis),
                         rec_dis).reshape(-1, 1)

    # Number of receivers
    nr = rec_locZ.size

    # Empty matrix for location of receivers and sources
    # including nr/ns row and two columns for x and z
    rec_loc = np.empty((nr, 2), dtype=np.float32)
    src_loc = np.empty((ns, 2), dtype=np.float32)

    if ns == 1:
        # Choosing the least possible x
        src_loc[:, 0] = (sdo/2+2)*dh  # offsetx/2 #

        # Specify the depth of source in the center of maximum depth
        src_loc[:, 1] = depth/2

    else:
        # Choosing the least possible x
        src_loc[:, 0] = (sdo/2+2)*dh

        # Distribut the depth of source
        src_loc[:, 1] = np.linspace(4*dh, depth-9*dh, ns)
        # src_loc[:,1]=np.linspace(4*dh, offset-9*dh, ns)

    rec_loc = np.hstack((np.ones((nr, 1))*offsetx-(sdo/2+2)*dh, rec_locZ))

    return src_loc.astype(np.float32), rec_loc.astype(np.float32)


def surface_seismic(ns, rec_dis, offsetx, dh, sdo):
    """
    A function to design a surface seismic acquisition

    Parameters
    ----------
        ns : int
            Number of sources

        rec_dis : float32
            Distance between receivers

        offsetx : flloat32
            Length of survey in x-direction

        depth : float32
            Depth of survey

        dh : float32
            Spatial sampling rate

        sdo: {2, 4, 8}
            Spatial order of finite difference method

    Returns
    -------

        src_loc: float32
            Location of sources
        rec_loc: float32
            Location of receivers
    """
    # Distribution of receivers
    rec_locX = np.arange(2*rec_dis, offsetx-2*rec_dis, rec_dis).reshape(-1, 1)

    # Number of receivers
    nr = rec_locX.size

    # Empty matrix for location of receivers and sources
    # including nr/ns row and two columns for x and z
    rec_loc = np.empty((nr, 2), dtype=np.float32)
    src_loc = np.empty((ns, 2), dtype=np.float32)

    if ns == 1:
        # Choosing the least possible z
        src_loc[:, 1] = (sdo/2+2)*dh  # depth/2 #

        # Specify the depth of source in the center of length in x-direction
        src_loc[:, 0] = offsetx/2
    else:
        # Choosing the least possible z
        src_loc[:, 1] = (sdo/2+2)*dh

        # Distribut sources in x-derection
        src_loc[:, 0] = np.linspace(4*dh, offsetx-9*dh, ns)

    rec_loc = np.hstack((rec_locX, np.ones((nr, 1))*(sdo/2+2)*dh))

    return src_loc, rec_loc


class Source:
    """
    A class for defining different types of sources.

    Parameters
    ----------
        src_loc : float32
            location of sources.

        dh : float
           Spatial sampling rate.
           
        dt: float
            Temporal sampling rate
        src_type : int, optional
            Source type: 0: explosive 
                         1: directional x
                         2: directional z
    """
    def __init__(self, src_loc, dh, dt, src_type=0):
        self.dh = dh
        self.i = np.int32(src_loc[:, 0]/self.dh)
        self.j = np.int32(src_loc[:, 1]/self.dh)
        self.dt = dt
 
        self.component = np.zeros(5, dtype=np.float32)
        if src_type == 0:
            self.component[2:4] = np.float32(1)
        elif src_type == 1:
            self.component[0] = np.float32(1)
        elif src_type == 2:
            self.component[1] = np.float32(1)
        else: 
            raise ('Please choose the right source type,\
                  either explosive (src_type = 0) or directional (src_type in [0, 1]')
                       
    def __call__(self, ind=None):
        
        if ind < self.w.size:
            return self.w[ind] * self.component
        else:
            return np.float32(0.0) * self.component

    def Ricker(self, fdom):
        """
        Amethod to generate Ricker wavelet.

        Parameters
        ----------
            fdom: float32
                Dominant frequency of wavelet

        """
        
        self.t = np.arange(-1.0/fdom, 1.0/fdom + self.dt/3, self.dt)
        self.w = np.float32((1.0 - 2.0*(np.pi*fdom*self.t)**2) * \
            np.exp(-(np.pi*fdom*self.t)**2))

    def delta(self):
        """
        A method to generate spike.

        Parameters
        ----------
            fdom: float32
                Dominant frequency of wavelet

        """
        
        self.w = np.float32(np.array([1]))
    
    
def acquisition_plan(ns, nr, src_loc, rec_loc, acq_type, n_well_rec, dh):
    """
    acquisition_plan generates the matrix of acquisition plan

    [extended_summary]

    Args:
        ns ([type]): [description]
        nr ([type]): [description]
        src_loc ([type]): [description]
        rec_loc ([type]): [description]
        acq_type ([type]): [description]
        n_well_rec ([type]): [description]
        dh ([type]): [description]

    Returns:
        [type]: [description]
    """
    data_guide = np.zeros((6, nr * ns))
    data_guide[0, :] = np.kron(np.arange(ns), np.ones(nr))
    data_guide[1, :] = np.kron(src_loc[:, 0] * dh, np.ones(nr))
    data_guide[2, :] = np.kron(src_loc[:, 1] * dh, np.ones(nr))
    data_guide[3, :] = np.kron(np.ones(ns), rec_loc[:, 0])
    data_guide[4, :] = np.kron(np.ones(ns), rec_loc[:, 1])
    
    if acq_type == 2:
        data_guide[4, :int(n_well_rec)] = np.flip(data_guide[4, :int(n_well_rec)])
    data_guide[5, :] = np.abs(data_guide[1, :] - data_guide[3, :])
        
    return data_guide


def discretized_acquisition_plan(data_guide, dh, npml=0):
    """
    discretized_acquisition_plan discretizes the matrix of acquisition plan

    [extended_summary]

    Args:
        data_guide ([type]): [description]
        dh ([type]): [description]
        npml (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    data_guide_sampling = np.copy(data_guide)
    data_guide_sampling[1:, :] = np.int32(data_guide_sampling[1:, :] / dh)
    data_guide_sampling[1:5, :] += npml
    data_guide_sampling = data_guide_sampling.astype(np.int32)
    
    return data_guide_sampling

def seismic_section(seismo, components=0, shape='3d'):
    seis_plan = {
        # 0 for (taux + tauz) / 2
        '1': ['taux'],
        '2': ['vx', 'vz'],
        '3': ['taux', 'tauz', 'tauxz'],
        '4': ['vx', 'vz', 'taux', 'tauz', 'tauxz']
    }
    data = {}
    
    if components != 0:
        for param in seis_plan[str(components)]:
            data[param] = seismo[param]
    else:
        data['taux'] = (seismo['taux'] + seismo['tauz']) / 2
        data['tauz'] = (seismo['taux'] + seismo['tauz']) / 2
    if shape == '2d':
        (nt, nr, ns) = data[[*data][0]].shape
        for par in data:
            data[par] = np.reshape(data[par], (nt, nr * ns), 'F')
    return data