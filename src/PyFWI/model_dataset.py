import numpy as np
import matplotlib.pyplot as plt

class ModelGenerator():

    def __init__(self, width, height, dx=1.0, dz=1.0):
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
        self.nx = np.int(width // dx)
        self.nz = np.int(height // dz)

    def background(self, bp):
        """
        add_layer genearte a layer of property.

        This method generates one layer with property "bp"

        Args:
            bp (dict): Background property
        """
        model = np.empty((self.nz, self.nx), dtype=np.float32)
        model[:, :] = bp

        # for depth in multilayers:

        return model

    def circle(self, bp, circle_prop, center, radius):
        """
        circle Provides a medium with  acircle inside it.

        This method generates the known circle model in the FWI studies. 

        Args:
            bp (float): Background property
            circle (flaot): Circle property
            radius (float): radius
            center (array): Center of circle as [x0, z0]
            
        """
        cx, cz = [center[0]//self.dx, center[1]//self.dz]
        radius = radius// self.dx
        model = {}
        
        model = self.background(bp)
        
        model = add_circle(model, circle_prop, radius, cx, cz)

        return model

    def add_layer (self, model, property, lt, lb, rt=None, rb=None):
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

        rt = (rt, [self.nx])[rt is None]

        model[lt[1]:lb[1], lt[0]:rt[0]] = property
        return model


    def add_anomaly(self, model, anomaly, x, z, width, height, type="circle"):
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
            r = (height // 2)/self.dx
            model = add_circle(model, anomaly, r, x//self.dx, z//self.dz)

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



        

if __name__ == "__main__":
    Model = ModelGenerator(1000, 1000, 10, 10)
    model = Model.circle(2500, 3000, [500,500], 100)

    model = Model.add_anomaly(model, 2800, 100, 100, 100, 200)
    model = Model.add_layer(model, 200, [0,50], [0,55])
    fig = plt.figure()
    im = plt.imshow(model)
    fig.colorbar(im)
    plt.show()