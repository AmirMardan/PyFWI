


def AcqParameters(ns, rec_dis, offsetx, offestz, dh, sdo, acq_type):
    """
    A function to define the acquisition based on user's demand

    Parameters
    ----------
        INPA : dictionary
            A dictionnary containing required parameters for iversion, at least:
                - ns: Number of sources
                - rec_dis: Distance between receivers
                - offsetx: Length of acquisition in x-direction
                - offsetz: Depth of acquisition
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

    if acq_type == 0:
        # In case of crosswell seismic

        src_loc, rec_loc = crosswell(ns, rec_dis, offsetx, offsetz,
                         dh, sdo)
        n_surface_rec = rec_loc.shape[0]
        n_well_rec = 0
    elif acq_type == 1:
        src_loc, rec_loc = SurfaceSeismic(ns, rec_dis, offsetx, offsetz,
                              dh, sdo)

        n_surface_rec = rec_loc.shape[0]
        n_well_rec = 0
    elif acq_type == 2:
        src_loc, rec_loc = SurfaceSeismic(ns, rec_dis, offsetx, offsetz,
                                          dh, sdo)
        n_surface_rec = rec_loc.shape[0]

        _, rec_loc2 = crosswell(ns, rec_dis, offsetx, offsetz,
                                     dh, sdo)

        rec_loc1_cst = np.ones((rec_loc2.shape), np.float32) @\
                   (np.array([[rec_loc[0, 1]], [0]], np.float32))
        rec_loc1 = np.hstack((rec_loc1_cst, rec_loc2[:, 1].reshape(-1,1)))
        rec_loc = np.vstack((rec_loc1, rec_loc, rec_loc2))
        n_well_rec = rec_loc2.shape[0]

    return src_loc, rec_loc, n_surface_rec, n_well_rec
