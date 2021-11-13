import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import add
import PyFWI.rock_physics as rp
import hdf5storage as hdf5
from PyFWI import seismic_io as io
from scipy.ndimage.filters import gaussian_filter
import os

class Circular():
    def __init__(self, vintage):
        self.vintage = vintage

    def louboutin(self):
        # louboutin et al., 2018, FWI, part2
        self.nx = self.nz = 100
        model = background((100, 100), {'vp': 2500.0})
        if self.vintage !=0:  # Not m0
            model['vp'] = add_circle(model['vp'], 3000, r=10, cx=50, cz=50)
        if self.vintage == 2:  # Monitor model
            model['vp'][25:30, 30: 41] -= 0.2 * model['vp'][25:30, 30: 41]
        return model

    def perturbation_pcs(self, smoothing=False):
        # Based on Hu et al., 2020, Direct rock physics inversion
        # Make another one based on DV
        model = background((100, 100), {'phi':0.2, 'cc':0.2, 'sw':0.8})
        if self.vintage !=0:  # Not m0
            model['phi'] = add_circle(model['phi'], 0.3, r=6, cx=25, cz=25)
            model['cc'] = add_circle(model['cc'], 0.4, r=6, cx=50, cz=50)
            model['sw'] = add_circle(model['sw'], 0.2, r=6, cx=75, cz=75)
        if self.vintage == 2:  # Monitor model
            model['sw'] = add_circle(model['sw'], 0.8, r=6, cx=75, cz=75)
        
        return model

class Laminar():
    def __init__(self, vintage) -> None:
        self.vintage = vintage

    def Hu_laminar(self, smoothing=None):
        # Based on Hu et al., 2020, Direct rock physics inversion 
        
        model = background((50, 50), {'phi':0.3, 'cc':0.1, 'sw':0.8})
        model = add_layer(model, {'phi':0.1, 'cc':0.5, 'sw':0.8}, [0, 35], [0, 50])
        model = add_layer(model, {'phi':0.2, 'cc':0.3, 'sw':0.8}, [0, 17], [0, 35])
            
        if self.vintage == 2:
            model = add_layer(model, {'sw':0.2}, [22, 17], [22, 25], rt=[28, 17])
        
        if smoothing:
            model = model_smoother(model, smoothing)
        return model


    def dupuy(self, smoothing):
        # based on Dupuy et al, 2016, 
        # Estimation of rock physics properties from seismic attributes — Part 2: Applications
        a = os.getcwdb()
        path = os.path.dirname(__file__) + '/data/Dupuy2011_Moduli_data.mat'
        # path = "src/PyFWI/data/Dupuy2011_Moduli_data.mat"
        # path = os.path.dirname(path)
        model ={}
        model = io.load_mat(path)

        if self.vintage == 1:
            model['s_gas'] *= 0
        
        model['k_d'], model['g_d'] = rp.drained_moduli(model['phi'], model['k_s'], model['g_s'], model['cs'])
        model['k_f'], model['rho_f'] = rp.voigt_berie(model['k_l'], model['rho_l'], model['k_gas'], model['rho_gas'], model['s_gas'])
        
        model['rho'] = rp.effective_density(model['phi'], model['rho_f'], model['rho_s'])
        model['k_u'], model['C'], model['M'] = rp.biot_gassmann(model['phi'], model['k_f'], model['k_s'], model['k_d'])

        model['vp'] = np.sqrt((model['k_u'] + 4 / 3 * model['g_d']) / model['rho'])
        model['vs'] = np.sqrt(model['g_d'] / model['rho'])

        model['vp'], model['vs'] = model['vp'] * np.sqrt(1e9), model['vs'] * np.sqrt(1e9)
        model["rho"] = model["rho"]/1000

        model['s_w'] = 1 - model['s_gas']

        for param in model:
            model[param] = model[param].astype(np.float32)

        if smoothing:
            model = model_smoother(model, smoothing)
        return model


class ModelGenerator(Circular, Laminar):

    def __init__(self, width=None, height=None, dx=1.0, dz=1.0, vintage=1):
        Circular.__init__(self, vintage)
        Laminar.__init__(self, vintage)
        """
        A class to create the synthetic model.

        This calss contain different moudulus to generate different types of synthetic models.

        Args:
            width (float):  Width of the model
            height (float): Depth of the model
            dx (float, optional): Spatial sampling rate in x-direction. Defaults to 1 (for importing the other parameters as number of samples).
            dz (float, optional): Spatial sampling rate in z-direction. Defaults to 1 (for importing the other parameters as number of samples).
        """
        self.width = width
        self.height = height
        self.dx = dx
        self.dz = dz
        # self.nx = np.int(width // dx)
        # self.nz = np.int(height // dz)


    def __call__(self, name, smoothing=False):
        # name of the model
        model = eval("self."+name)(smoothing)
        return model 


def add_anomaly(model, anomaly, x, z, dx, dz, height, type="circle"):
    """
    add_anomaly adds anomaly to the previously created model.

    This mathod add an anomally to the Earth mode that is already createad.

    Args:
        model (float): The previously created model. 
        anomaly (float): The properties of the anomaly
        x ([type]): x-location of the anomaly
        z ([type]): z-location of the anomaly
        width ([type]): Width of the anomaly
        height ([type]): Height of the anomaly
        type (str, optional): The shape of the anomaly. Defaults to "circle".

    Returns:
        model (dict): The new model.
    """

    if type in ["circle", "Circle"]:
        r = (height // 2)/dx
        model = add_circle(model, anomaly, r, x//dx, z//dz)

    return model


def background(size, params):
    """
    add_layer genearte a layer of property.

    This method generates one layer with property "bp"

    Args:
        bp (dict): Background property
    """
    (nz, nx) = size
    model = {}
    for param in params:
        model[param] = np.empty((nz, nx), dtype=np.float32)
        
        model[param][:, :] = params[param]

    return model

    # def circle(self, bp, circle_prop, center, radius):
    #     """
    #     circle Provides a medium with  acircle inside it.

    #     This method generates the known circle model in the FWI studies. 

    #     Args:
    #         bp (float): Background property
    #         circle (flaot): Circle property
    #         radius (float): radius
    #         center (array): Center of circle as [x0, z0]
            
    #     """
    #     cx, cz = [center[0]//self.dx, center[1]//self.dz]
    #     radius = radius// self.dx
    #     model = {}
        
    #     model = self.background(bp)
        
    #     model = add_circle(model, circle_prop, radius, cx, cz)

    #     return model


def add_layer (model, property, lt, lb, rt=None, rb=None):
        """
        add_layer add alyer to the model

        This function add a layer to the mdoel

        Args:
            model (dict): Already created model.
            property (dict): Property of the new layer
            lt (array, int): Sample number ([x ,z]) of the top of the layer in the most left part
            lb (array, int): Sample number ([x ,z]) of the bottom of the layer in the most left part
            rt (array, int): Sample number ([x ,z]) of the top of the layer in the most right part
            rb (array, int): Sample number ([x ,z]) of the bottom of the layer in the most right part #TODO: to develop for dipping layers

        Returns:
            model(dict): Return the model.
        """
        nx = model[[*model][0]].shape[1]
        rt = (rt, [nx])[rt is None]

        for param in property:
            model[param][lt[1]:lb[1], lt[0]:rt[0]] = property[param]
        return model

def add_circle (model, circle_prop, r, cx, cz):
    """
    add_circle adds a circle to the model

    This function generates a circle in the model.

    Args:
        model (float): Already created model.
        circle_prop (float): Property of the circle.
        r (int): Radius of the circle 
        cx (int): x_location of the center
        cz (int): z-location of the center

    Returns:
        model(dict): Return the model.
    """
    [nz, nx] = model.shape

    for i in range(nz):
        for j in range(nx):
            if (i-cz)**2+(j-cx)**2 < r ** 2:
                model[i, j] = circle_prop

    return model


def model_smoother(model, smooting_value):
    for params in model:
        model[params] = gaussian_filter(model[params], smooting_value)
    return model

def Hu_circle(rho=None, prop_back=None, prop_circle=None, nz=100, nx=100, r=8, monitor=False):
    # GET the model if we are looking for monitor model 

    if not rho:
        rho = {
            "Q": 2.65,
            "Clay": 2.55,
            "Water": 1.0,
            "hydro": 0.1
            }

    if not prop_back:
        prop_back = {
            "phi": 0.2,
            'cc': 0.2,
            'sw': 0.4
        }

    if not prop_circle:
        prop_circle = {
            "phi": 0.3,
            'cc': 0.4,
            'sw': 0.2
        }
         
    if monitor: # For monitor model, sw decreases
        if not prop_circle:
            prop_circle['sw'] = 0.8
            

    Model = ModelGenerator(nz, nx, 1, 1)

    loc = np.array([nz//4, nx//4])

    model = {'phi': Model.circle(prop_back['phi'], prop_circle['phi'], loc, r),
             'cc': Model.circle(prop_back['cc'], prop_circle['cc'], 2*loc, r),
             'sw': Model.circle(prop_back['sw'], prop_circle['sw'], 3*loc, r)

    }

    rho_m = rp.Density().matrix(rho['Clay'], model['cc'], rho['Q'])
    rho_f = rp.Density().fluid(rho['hydro'], rho['Water'], model['sw'])

    vp, vs = rp.Han(model['phi'], model['cc'], a1=5.77, a2=6.94, a3=1.728, b1=3.7, b2=4.94, b3=1.57)

    model['vp'] = 1000 * vp
    model['vs'] = 1000 * vs
    model['rho'] = (1-model['phi']) * rho_m + model['phi'] * rho_f

    return model
        

if __name__ == "__main__":
    Model = ModelGenerator()
    model = Model('Hu_laminar', False)
    
    Model = Laminar(vintage=2)
    model = Model.dupuy(smoothing=False)
    fig = plt.figure()
    # phi = model['phi']
    im = plt.imshow(model['vp'])
    fig.colorbar(im)
    plt.show()