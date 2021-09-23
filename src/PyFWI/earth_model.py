
import numpy as np
import matplotlib.pyplot as plt

class Model_generator():

    def __init__(self, width, height, dx=1.0, dz=1.0):
        """
        A class to create the synthetic model.

        This calss contain different moudulus to generate different types of synthetic models.

        Args:
            width (float):  Width of the model
            height (float): Depth of the model
            dx (float, optional): Spatial sampling rate. Defaults to 1 (for importing the other parameters as number of samples).
            dz (float, optional): [description]. Defaults to 1 (for importing the other parameters as number of samples).
        """
        self.width = width
        self.height = height
        self.dx = dx
        self.dz = dz
        self.nx = width // dx
        self.nz = height //dz

    def circle(self, bc, circle_prop, center, radius):
        """
        circle Provides a medium with  acircle inside it.

        This method generates the known circle model in the FWI studies. 

        Args:
            bc (dict): Background property
            circle (dict): Circle property
            radius (float): radius
            center (array): Center of circle as [x0, z0]
            
        """
        cx, cz = [center[0]//self.dx, center[1]//self.dz]
        radius = radius// self.dx
        model = {}
        for params in bc:
            model[params] = np.empty((self.nz, self.nx), dtype=np.float32)
            model[params][:, :] = bc[params]

        model = add_circle(model, circle_prop, radius, cx, cz)

        return model

    def add_anomaly(self, model, anomaly, x, z, width, height, type="circle"):
        """
        add_anomaly adds anomaly to the previously created model.

        This mathod add an anomally to the Earth mode that is already createad.

        Args:
            model (dict): The previously created model. 
            anomaly (dict): The properties of the anomaly
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
    add_circle [summary]

    [extended_summary]

    Args:
        model (dict): Already created model.
        circle_prop (dict): Property of the circle.
        r (int): Radius of the circle 
        cx (int): x_location of the center
        cz (int): z-location of the center

    Returns:
        model(dict): Return the model.
    """
    [nz, nx] = model[list(model.keys())[0]].shape

    for i in range(nz):
        for j in range(nx):
            if (i-cz)**2+(j-cx)**2 < r ** 2:
                for params in model:
                    model[params][i, j] = circle_prop[params]

    return model
        

if __name__ == "__main__":
    Model = Model_generator(1000, 1000, 10, 10)
    model = Model.circle({"vp":2500}, {'vp': 3000}, [500,500], 100)

    model = Model.add_anomaly(model, {'vp':2800}, 100, 100, 100, 200)

    fig = plt.figure()
    im = plt.imshow(model['vp'])
    fig.colorbar(im)
    plt.show()