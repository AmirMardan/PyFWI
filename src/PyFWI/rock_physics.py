import numpy as np
from numpy.core.shape_base import block


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

    
    def fluid(self, r_hydro, rho_w, sw):
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

    def matrix(self, rho_clay, cc, rho_q, **kwargs):
        """
        matrix [summary]

        [extended_summary]

        Args:
            rho_clay ([type]): [description]
            cc ([type]): [description]
            rho_q ([type]): [description]

        Returns:
            [type]: [description]
        """
        q = 1 - cc 
        rho_m = rho_clay * cc + rho_q * q

        return rho_m


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
    

def Han(phi=None, cc=None, vp=None, vs=None, a1=5.77, a2=6.94, a3=1.728, b1=3.70, b2=4.94, b3=1.57):
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
    if phi is not None: # To calculate p_wave (p1) and s_wave (p2) velocities
        p1 = a1 - a2 * phi - a3 * np.sqrt(cc) 
        p2 = b1 - b2 * phi - b3 * np.sqrt(cc)

        # p2 = p2.astype(np.float32)
        # p1 = p1.astype(np.float32)
        # p1 *= 1000
        # p2 *= 1000
    elif vp is not None:
        original_shape = np.shape(vp)

        vp = np.copy(vp)#/1000
        vs = np.copy(vs)#/1000

        vp = vp.reshape(1, -1)
        vs = vs.reshape(1, -1)

        y1 = vp - a1
        y2 = vs - b1
        y = np.vstack((y1, y2))

        n = vp.shape[0]
        A1 = np.hstack((-a2*np.ones((n, n)), -a3*np.ones((n, n))))
        A2 = np.hstack((-b2*np.ones((n, n)), -b3*np.ones((n, n))))
        A = np.vstack((A1, A2))

        p = np.linalg.solve(A, y)

        p1 = p[0, :]
        p2 = (p[1, :]) ** 1
        # print(p2) 

        p1 = p1.reshape(original_shape)
        p2 = p2.reshape(original_shape)

        p1[p1<0] = 0
        p2[p2<0] = 0
    return np.round(p1,2) , np.round(p2, 2)


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


def effective_density(phi, rho_f, rho_s):
    return rho_f * phi + rho_s * (1 - phi)


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

    # vp1= np.array([model['vp'][50,50]])/1000
    # vs1= np.array([model['vs'][50,50]])/1000
    
    print("VP: {}, VS: {}".format(vp1, vs1))
    phi1, cc1 = Han(vp=vp1, vs=vs1)

    print("phi: {}, cc: {}".format(phi1, cc1))
    
    a=1
    # print("=================")
    # evp, evs = Han(phi=phi, cc=cc)

    # print(evp*1000)
    # print(vp)
    # print(" ------- ")
    # print(evs*1000)
    # print(vs)

    # im = plt.imshow(cc)
    # plt.colorbar()
    # plt.show()
    