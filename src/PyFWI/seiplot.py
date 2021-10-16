import matplotlib.pyplot as plt
import matplotlib as mlp
from numpy.core.shape_base import block
import numpy as np

import plotly.express as px

try:
    from PyFWI.model_dataset import ModelGenerator
except:
    from model_dataset import ModelGenerator

def earth_model(model, keys=[]):

    n = len(model)
    fig = plt.figure()

    i = 1

    if keys == []:
        params= model.keys()

    for param in params:
        ax = fig.add_subplot(1,n,i)
        aspect = (model[param].shape[0]/model[param].shape[1])  

        offsetx =100; depth=100
        ax.axis([0, offsetx, 0, depth])
        ax.set_aspect(aspect)

        im = ax.imshow(model[param])
        i +=1
        fig.colorbar(im, ax=ax, shrink=aspect+0.1,
                        pad=0.01, label=param)
    

def seismic_section(ax, data, x_axis=None, t_axis=None, aspect_preserving=False):
    if aspect_preserving:
        aspect = (data.shape[0]/data.shape[1])
        ax.set_aspect(aspect)

    if not x_axis:
        x_axis = np.arange(data.shape[1])
    
    if not t_axis:
        t_axis = np.arange(data.shape[0])

    im = ax.pcolor(x_axis, t_axis, data, vmax=data.max(), vmin=data.min(),  cmap='gray', shading='nearest')

    ax.invert_yaxis()
    ax.axis([0, x_axis[-1], t_axis[-1], 0])
    plt.show(block=False)

    return ax

if __name__ == "__main__":
    import model_dataset as md
    import rock_physics as rp

    [nz, nx] = [100, 100]
    Model = ModelGenerator(nx, nz, 1, 1)
    model = Model.circle({"vp":2500}, {"vp": 3000}, [50, 50], 10)
    model = rp.Density().gardner(model)

    earth_model(model)
    plt.show()
    print(4)
