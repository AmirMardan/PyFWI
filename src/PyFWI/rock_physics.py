import numpy as np
from numpy.core.shape_base import block


class Density:
    def __init__(self):
        pass

    def __call__(self, rp_model):
        pass

    def gardner (self, model, units="metric"):
        
        coeff = (0.31, 0.23) [units in ['imperial', 'Imperial', 'ft', 'ft/s']]
        model['rho'] = 0.31 * model['vp'] ** 0.25


    


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import earth_model as em


    Model = em.Model_generator(100, 100, 1, 1)
    model = Model.layer({"vp": 2500}) 
    model = em.add_layer(model, {'vp':2400}, [0, 50], [0, 70], [100, 50] )

    y = Density()
    
    y.gardner(model, "metric")
    
    im = plt.imshow(model["vp"])
    plt.colorbar()
    plt.show()
    