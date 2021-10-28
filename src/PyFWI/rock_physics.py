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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import earth_model as em


    Model = em.Model_generator(100, 100, 1, 1)
    model = Model.layer({"vp": 2500}) 
    model = em.add_layer(model, {'vp':2400}, [0, 50], [0, 70], [100, 50] )

    model = Density().gardner(model, "metric")
    model = ShearVelocity().poisson_ratio_vs(model, 0.25)
    
    im = plt.imshow(model["vs"])
    plt.colorbar()
    plt.show()
    