import matplotlib.pyplot as plt
import matplotlib as mlp
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numpy.core.shape_base import block
import numpy as np

def earth_model(model, keys=[],offset=None, depth= None, **kwargs):
    """
    earth_model show the earth model.

    This function is provided to show the earth models.

    Args:
        model (Dictionary): A dictionary containing the earth model.
        keys (list, optional): List of parameters you want to show. Defaults to [].

    Returns:
        fig (class): The figure class  to which the images are added for furthur settings like im.set-clim(). 
    """
    nx = max(model[[*model][0]].shape[1], model[[*model][0]].shape[1])
    nz = max(model[[*model][0]].shape[0], model[[*model][0]].shape[0])
    if offset is None:
        offset = nx
        
    if depth is None:
        depth = nz
        
    if keys == []:
        keys = model.keys()
        
    n = len(keys)
    fig = plt.figure(figsize=(4*n, 4))

    i = 1
    ims = []

    for param in keys:
        ax = fig.add_subplot(1, n, i)
        aspect = (model[param].shape[0]/model[param].shape[1])  

        ax.axis([0, offset, 0, depth])
        ax.set_aspect(aspect)

        im = ax.imshow(model[param], **kwargs)
        ims.append(im)
        axes_divider = make_axes_locatable(ax)
        cax = axes_divider.append_axes('right', size='7%', pad='2%')
        
        fig.colorbar(im, cax=cax, shrink=aspect+0.1,
                        pad=0.01)
        ax.invert_yaxis()
        ax.set_title(param, loc='center')
        if i>1:
            ax.set_yticks([])
        i +=1
    fig.__dict__['ims'] = ims
    
    return fig
    

def seismic_section(ax, data, x_axis=None, t_axis=None, aspect_preserving=False, **kargs):
    """
    seismic_section show seismic section


    Parameters
    ----------
    ax : Axes or array of Axes
        ax to arrange plots
    data : float
        Seismic section
    x_axis : array
        X-axis, by default None
    t_axis : array, optional
        t-axis, by default None
    aspect_preserving : bool, optional
        _description_, by default False

    Returns
    -------
    ax : Axis
        The ax for more adjustment
    """
    if aspect_preserving:
        aspect = (data.shape[0]/data.shape[1])
        ax.set_aspect(aspect)

    if x_axis is None:
        x_axis = np.arange(data.shape[1])
    
    if t_axis is None:
        t_axis = np.arange(data.shape[0])

    im = ax.pcolor(x_axis, t_axis, data,  cmap='gray', **kargs)

    ax.invert_yaxis()
    ax.axis([0, x_axis[-1], t_axis[-1], 0])
    # plt.show(block=False)

    return ax

def gn_plot(p, grad, nz, nx):
    n_elemetns = nz * nx
    
    fig = plt.figure()
    ax = fig.add_subplot(3, 2, 1)
    im = ax.imshow(p[:n_elemetns].reshape(nz,nx), cmap='jet')
    axes_divider = make_axes_locatable(ax)
    cax = axes_divider.append_axes('right', size='7%', pad='2%')
    cb = fig.colorbar(im, cax=cax)
    ax = fig.add_subplot(3, 2, 2)
    im = ax.imshow(-grad[:n_elemetns].reshape(nz,nx), cmap='jet')
    axes_divider = make_axes_locatable(ax)
    cax = axes_divider.append_axes('right', size='7%', pad='2%')
    cb = fig.colorbar(im, cax=cax)
                
    ax = fig.add_subplot(3, 2, 3)
    im = ax.imshow(p[n_elemetns:2*n_elemetns].reshape(nz,nx), cmap='jet')
    axes_divider = make_axes_locatable(ax)
    cax = axes_divider.append_axes('right', size='7%', pad='2%')
    cb = fig.colorbar(im, cax=cax)
    ax = fig.add_subplot(3, 2, 4)
    im = ax.imshow(-grad[n_elemetns:2*n_elemetns].reshape(nz,nx), cmap='jet')
    axes_divider = make_axes_locatable(ax)
    cax = axes_divider.append_axes('right', size='7%', pad='2%')
    cb = fig.colorbar(im, cax=cax)
                
    ax = fig.add_subplot(3, 2, 5)
    im = ax.imshow(p[2*n_elemetns:].reshape(nz,nx), cmap='jet')
    axes_divider = make_axes_locatable(ax)
    cax = axes_divider.append_axes('right', size='7%', pad='2%')
    cb = fig.colorbar(im, cax=cax)
    ax = fig.add_subplot(3, 2, 6)
    im = ax.imshow(-grad[2*n_elemetns:].reshape(nz,nx), cmap='jet')
    axes_divider = make_axes_locatable(ax)
    cax = axes_divider.append_axes('right', size='7%', pad='2%')
    cb = fig.colorbar(im, cax=cax)

def inversion_video(m_video, pause=0.2, **kwargs):
    """
    inversion_video show the procedure of updating the model during the FWI

    Args:
        m_video ([type]): Matrix of model after each iteration
        pause (float, optional): Lenght of pause between each model. Defaults to 0.2.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    axes_divider = make_axes_locatable(ax)
    cax = axes_divider.append_axes('right', size='4%', pad='2%')
        
    for i in range(m_video.shape[2]):
        im = ax.imshow(m_video[20:-20, 20:-20, i], **kwargs)
        cb = fig.colorbar(im, cax=cax)
        ax.set_title("Iteration {}".format(i + 1))
        plt.pause(pause)
        
        
        
if __name__ == "__main__":
    # import PyFWI.model_dataset as md
    import PyFWI.rock_physics as rp
    from PyFWI.model_dataset import ModelGenerator

    [nz, nx] = [100, 100]
    Model = ModelGenerator(nx, nz, 1, 1)
    model = {
        'vp': Model.circle(2500, 3000, [50, 50], 10)
    }
    model['rho'] = rp.Density().gardner(model['vp'])

    earth_model(model)
    plt.show()
    print(4)

