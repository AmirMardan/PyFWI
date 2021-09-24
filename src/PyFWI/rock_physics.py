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

    def gardner (self, model, units="metric"):
        """
        gardner method to estimate the density

        This mdethod estimate density of a model based on P-wave velocity. It
        uses the Gardner's eqution.

        Args:
            model (dict): A dictionary containg Earth's properties (must containg 'vp')
            units (str, optional): Specify the system of the units fo measurements (Metric or Imperial) . Defaults to "metric".
        """
        coeff = (0.31, 0.23) [units in ['imperial', 'Imperial', 'ft', 'ft/s']]
        model['rho'] = coeff * model['vp'] ** 0.25
    
        return model


class ShearVelocity:
    def __init__(self):
        pass

    def poisson_ratio_vs(self, model, sigma=None):
        """
        poisson_ratio_vs calculates the shear velocity.

        Calculates the shear velocity based on Poisson's ration.

        Args:
            model (dict): Model containing the P-wave velocity.
            sigma (float, optional): Poisson's ration. It could be None if parameter "model" has this property. Defaults to None.

        Returns:
            model (dict): The input model and shear velocity is added.
        """

        if sigma is None:
            try:
                sigma = model["poisson_ration"]
            except:
                raise NameError("To calculate the S-wave velocity based on Poisson's ratio, this parameter has to be imported.")

        model['vs'] = model['vp'] * np.sqrt((0.5 - sigma) / (1.0 - sigma))
        return model
    

class Mu:
    def __init__(self):
        pass

    def vs_rho(self, model): 
        """
        vs_rho generate mu

        This function add mu to to the imported model based on S-wave velocity and density.

        Args:
            model (dict): Model of Earth's properties which must contain vs (S-wave velocity) and rho (density).

        Returns:
            model: It returns the model back while mu is added.
        """
        model['mu'] = model['vs'] ** 2 * model['rho']

        return model



class Lame:
    def __init__(self):
        pass
    
    def vp_rho_mu(self, model):

        # TODO: CHECK IF THERE IS NO MU BUT MODEL HAS VS, CREATE MU BASED ON THE VS AND THEN GENERATE MU
        
        model['lam'] = model['rho'] * model['vp'] ** 2 - 2 * model['mu']
        return model



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
    