import numpy as np
from numpy.core.shape_base import block
from scipy.sparse import coo_matrix,linalg

class Density:
    def __init__(self):
        """
        A calss to estimate density.

        Density is a class to estimate the density based on rock-physical models.

        """
        pass

    def __call__(self, rp_model):
        pass

    def gardner (self, vp, units="metric"):
        """
        gardner method to estimate the density

        This mdethod estimate density of a model based on P-wave velocity. It
        uses the Gardner's eqution.

        Args:
            vp (float): P-wave velocity
            units (str, optional): Specify the system of the units fo measurements (Metric or Imperial) . Defaults to "metric".
        
        Returns:
            rho: density
        """
        coeff = (0.31, 0.23)[units in ['imperial', 'Imperial', 'ft', 'ft/s']]
        rho = coeff * vp ** 0.25
        return rho

    @staticmethod
    def effective_density(phi, rho_f, rho_s):
        return rho_f * phi + rho_s * (1 - phi)

    @staticmethod
    def fluid(r_hydro, rho_w, sw):
        """
        fluid [summary]

        [extended_summary]

        Args:
            r_hydro ([type]): [description]
            rho_w ([type]): [description]
            sw ([type]): [description]

        Returns:
            rho_f [type]: Density of fluid
        """
        sh = 1 - sw
        rho_f = rho_w * sw + r_hydro * sh

        return rho_f

    @staticmethod
    def matrix(rho_clay, cc, rho_q, **kwargs):
        """
        matrix [summary]

        [extended_summary]

        Args:
            rho_clay ([type]): [description]
            cc ([type]): [description]
            rho_q ([type]): [description]

        Returns
        --------
            rho_m: ndarray
                Density of matrix
        """
        q = 1 - cc 
        rho_m = rho_clay * cc + rho_q * q

        return rho_m

    
    def rho_from_pcs(self, rho_c, rho_q, rho_w, rho_g, cc, sw, phi):
        """
        This function calculate density from Porosity, clay content, and water Saturation

        Args:
            rho_c:
                Density of clay
            rho_q:
                Density of quartz
            rho_w:
                Density of water
            rho_g:
                Density of gas
            cc:
                clay content
            sw:
                water saturation
            phi:
                Porosity

        Returns
        --------
        rho: float
            Effective density
        """
        rho_s = self.matrix(rho_c, cc, rho_q)
        rho_f = self.fluid(rho_g, rho_w, sw)

        rho = self.effective_density(phi, rho_f, rho_s)
        return rho.astype(np.float32) 

class ShearVelocity:
    def __init__(self):
        pass

    def poisson_ratio_vs(self, vp, sigma=0.25):
        """
        poisson_ratio_vs calculates the shear velocity.

        Calculates the shear velocity based on Poisson's ration.

        Args:
            vp (float): P-wave velocity.
            sigma (float, optional): Poisson's ration. It could be None if parameter "model" has this property. Defaults to None.

        Returns:
            vs: The input model and shear velocity is added.
        """            
        vs = vp * np.sqrt((0.5 - sigma) / (1.0 - sigma))
        return vs

    def Han(self, phi, cc, **kwargs):
        """
        Han calulates vs based on Han empirical model.

        Han calulates vs based on Han empirical model.

        Args:
            phi ([type]): Porosity
            cc ([type]): Clay content

        Returns:
            vp: S-wave velocity
        """
        _, vs = Han(phi, cc, kwargs)
        return vs
    
    

class Mu:
    def __init__(self):
        pass

    def vs_rho(self, vs, rho=None): 
        """
        vs_rho generate mu

        This function add mu to to the imported model based on S-wave velocity and density.

        Args:
            vs (float or dict): S-wave velocity. if dict, it has to contain value for density.
            rho (float, option): Density

        Returns:
            mu: Shear modulus
        """
        if rho is None:
            try:
                rho = vs['rho']
                vs = vs['vs']
            except:
                error_lack_of_data

        mu = vs ** 2 * rho

        return mu



class Lamb:
    def __init__(self):
        pass
    
    def vp_rho_mu(self, rho, vp=None, mu=None):

        # TODO: CHECK IF THERE IS NO MU BUT MODEL HAS VS, CREATE MU BASED ON THE VS AND THEN GENERATE MU
        if vp is None or mu is None:
            try:
                vp = rho['vp']
                mu = rho['mu']
                rho = rho['rho']
            except:
                error_lack_of_data()

        lam = rho * vp ** 2 - 2 * mu
        return lam


def error_lack_of_data():
    raise "Not appropriate data are provided to calculate Lambda"


class p_velocity:
    def __init__(self):
        pass
    
    def lam_mu_rho(self, lam, mu, rho):        
        vp = np.sqrt((lam + 2*mu)/rho)
        return vp

    def Han(self, phi, cc, **kwargs):
        """
        Han calulates vp based on Han empirical model.

        Han calulates vp based on Han empirical model.

        Args:
            phi ([type]): Porosity
            cc ([type]): Clay content

        Returns:
            vp: P-wave velocity
        """
        vp, _ = Han(phi, cc)
        return vp

    def gardner(rho, units='metric'):
        coeff = (0.31, 0.23)[units in ['imperial', 'Imperial', 'ft', 'ft/s']]
        vp = (rho/coeff)**4
        return vp
    

def Han(phi, cc, a1=5.5, a2=6.9, a3=2.2, b1=3.4, b2=4.7, b3=1.8):
    """
    Han estimates velocity based on porosity and clasy content

    Han found empirical regressions relating ultrasonic (laboratory) velocities to porosity and clay content

    Args:
        phi ([type]): [porosity
        cc ([type]): clay content
        a1 (float, optional): Constant value for Vp. Defaults to 5.77.
        a2 (float, optional): Constant value for Vp. Defaults to 6.94.
        a3 (float, optional): Constant value for Vp. Defaults to 1.728.
        b1 (float, optional): Constant value for Vs. Defaults to 5.77.
        b2 (float, optional): Constant value for Vs. Defaults to 6.94.
        b3 (float, optional): Constant value for Vs. Defaults to 1.728.

    Returns:
        vp: P-wave velocity (km/s)
        vs = S-wave velocity (km/s)
    References:
        1. Hu et al, 2021, Direct updating of rock-physics properties using elastice full-waveform inversion
        2. Mavko, G., Mukerji, T., & Dvorkin, J., 2020, The rock physics handbook. Cambridge university press.
    """
    vp = a1 - a2 * phi - a3 * cc  # np.sqrt(cc) 
    vs = b1 - b2 * phi - b3 * cc  # np.sqrt(cc)

    vp = vp.astype(np.float32)
    vs = vs.astype(np.float32)
        
    return vp, vs


def reverse_Han(vp, vs, a1=5.5, a2=6.9, a3=2.2, b1=3.4, b2=4.7, b3=1.8):
    nz, nx = vp.shape
    n_elements = nz * nx
    
    
    vp0 = np.copy(vp)#/1000
    vs0 = np.copy(vs)#/1000
    
    a2 = a2 * np.ones((n_elements)) 
    a3 = a3 * np.ones((n_elements)) 
    b2 = b2 * np.ones((n_elements)) 
    b3 = b3 * np.ones((n_elements)) 
    
    y1 = a1 - vp0
    y2 = b1 - vs0
    
    b = np.hstack((y1.reshape(-1), y2.reshape(-1))) 
    
    a_diag = np.arange(2*n_elements)
    a_first = np.arange(n_elements)
    a_second = np.arange(n_elements, 2*n_elements)    
    
    row = np.hstack((a_diag, a_first, a_second))
    col = np.hstack((a_diag, a_second, a_first))
    data = np.hstack((a2, b3, a3, b2))
    
    a = coo_matrix((data, (row, col)), dtype=np.float32)

    # x = np.linalg.solve(a, b)
    x = linalg.spsolve(a, b)
    phi = x[:n_elements].reshape(nz, nx)
    cc = x[n_elements:].reshape(nz, nx)
    
    return phi , cc


def drained_moduli(phi, k_s, g_s, cs):
    if (phi >= 1).any():
        phi /= 100

    k_d = k_s * ((1 - phi) / (1 + cs * phi))

    g_d = g_s * ((1 - phi) / (1 + 1.5 * cs * phi))
    return k_d, g_d


def voigt_berie(k_l, rho_l, k_g, rho_g, s_g):
    k_f = (k_l - k_g) * ((1 - s_g) ** 5) + k_g
    rho_f = rho_l * (1 - s_g) + rho_g * s_g
    return k_f, rho_f


def biot_gassmann(phi, k_f, k_s, k_d):
    Delta = delta_biot_gassmann(phi, k_f, k_s, k_d)

    denom = phi * (1 + Delta)

    k_u = (phi * k_d + (1 - (1 + phi) * (k_d / k_s)) * k_f) / denom
    C = k_f * (1 - k_d / k_s) / denom
    M = k_f / denom
    return k_u, C, M


def delta_biot_gassmann(phi, k_f, k_s, k_d):
    if (phi >= 1).any():
        phi /= 100
    return ((1 - phi) / phi) * (k_f / k_s) * (1 - (k_d / (k_s - k_s * phi)))


def lmd2vd(lam, mu, rho):
    """
    lmd2vd switches Lama modulus and density to vp, vs, density

    [extended_summary]

    Args:
        lam ([type]): [description]
        mu ([type]): [description]
        rho ([type]): [description]

    Returns:
        [type]: [description]
    """
    vp = np.sqrt((lam + 2 * mu)/ rho)
    vs = np.sqrt((lam + 2 * mu)/ rho)
    rho = rho
    return vp, vs, rho


def vd2lmd(vp, vs, rho):
    """
    vd2lmd switches vp, vs, density to Lame modulus and density to 

    [extended_summary]

    Args:
        vp ([type]): [description]
        vs ([type]): [description]
        rho ([type]): [description]

    Returns:
        [type]: [description]
    """
    lam = rho * (vp ** 2 - 2 * vs ** 2)
    mu = rho * vs ** 2
    rho = rho
    return lam, mu, rho


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from  PyFWI import model_dataset as em


    Model = em.ModelGenerator(100, 100, 1, 1)
    model = {'vp': Model.background(2500)}

    model['vp'] = Model.add_layer(model['vp'], 2400, [0, 50], [0, 70], [100, 50] )

    model['rho'] = Density().gardner(model['vp'], "metric")
    model['vs'] = ShearVelocity().poisson_ratio_vs(model['vp'], 0.25)
    
    vp1, vs1 = Han(phi=0.55, cc=0.25)
    
    vp1 = np.array([vp1])
    vs1 = np.array([vs1])
    
    print("VP: {}, VS: {}".format(vp1, vs1))
    phi1, cc1 = Han(vp=vp1, vs=vs1)

    print("phi: {}, cc: {}".format(phi1, cc1))
    