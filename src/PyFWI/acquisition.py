from logging import exception
import numpy as np


def AcqParameters(ns, rec_dis, offsetx, depth, dh, sdo, acq_type):
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
        rec-loc: float32
            Location of receivers
    """
    if rec_dis < dh:
        raise Exception("Receiver distance should be larger than spatial sampling")
    if acq_type == 0:
        # In case of crosswell seismic

        src_loc, rec_loc = crosswell(ns, rec_dis, offsetx, depth,
                         dh, sdo)
        n_surface_rec = rec_loc.shape[0]
        n_well_rec = 0
    elif acq_type == 1:
        src_loc, rec_loc = SurfaceSeismic(ns, rec_dis, offsetx, depth,
                              dh, sdo)

        n_surface_rec = rec_loc.shape[0]
        n_well_rec = 0
    elif acq_type == 2:
        src_loc, rec_loc = SurfaceSeismic(ns, rec_dis, offsetx, depth,
                                          dh, sdo)
        n_surface_rec = rec_loc.shape[0]

        _, rec_loc2 = crosswell(ns, rec_dis, offsetx, depth,
                                     dh, sdo)

        rec_loc1_cst = np.ones((rec_loc2.shape), np.float32) @\
                   (np.array([[rec_loc[0, 1]], [0]], np.float32))
        rec_loc1 = np.hstack((rec_loc1_cst, rec_loc2[:, 1].reshape(-1,1)))
        rec_loc = np.vstack((rec_loc1, rec_loc, rec_loc2))
        n_well_rec = rec_loc2.shape[0]

    return src_loc, rec_loc, n_surface_rec, n_well_rec


def crosswell(ns, rec_dis, offsetx, offsetz,
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

        offsetz : float32
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
    rec_locZ = np.arange(2*rec_dis, offsetz-(2*rec_dis),
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
        src_loc[:, 1] = offsetz/2

    else:
        # Choosing the least possible x
        src_loc[:, 0] = (sdo/2+2)*dh

        # Distribut the depth of source
        src_loc[:, 1] = np.linspace(4*dh, offsetz-9*dh, ns)
        # src_loc[:,1]=np.linspace(4*dh, offset-9*dh, ns)

    rec_loc = np.hstack((np.ones((nr, 1))*offsetx-(sdo/2+2)*dh, rec_locZ))

    return src_loc.astype(np.float32), rec_loc.astype(np.float32)


def SurfaceSeismic(ns, rec_dis, offsetx, offsetz,
                   dh, sdo):
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

        offsetz : float32
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
        src_loc[:, 1] = (sdo/2+2)*dh  # offsetz/2 #

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
    """
    def __init__(self, src_loc, dh, dt):
        self.dh = dh
        self.i = np.int32(src_loc[:, 0]/self.dh)
        self.j = np.int32(src_loc[:, 1]/self.dh)
        self.dt = dt
        
    def __call__(self, ind=None):
        if ind < self.w.size:
            return self.w[ind], self.w[ind]
        else:
            return np.float32(0.0), np.float32(0.0)

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
        Amethod to generate Ricker wavelet.

        Parameters
        ----------
            fdom: float32
                Dominant frequency of wavelet

        """
        
        self.w = np.float32(np.array([1]))
