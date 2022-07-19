import numpy as np
import matplotlib.pyplot as plt

from PyFWI.seiplot import seismic_section

class Gain():

    def __init__(self, t, dt, nt):
        """
        A class to create gain

        This class is provided to generate different gain function

        Args:
            t (float): The total time of the data
            dt (float): Sampling rate
            nt (int): Number of time sample
        """
        self.t = t
        self.dt = dt
        self.nt = nt
    
    def time_linear(self, show_gain=False):
        """
        time_linear generates a linear gain function related to time

        This function generates a linear gain function with time

        Args:
            show_gain (bool, optional): If we need to plot the gain function. Defaults to False.

        Returns:
            GF (class): A class for applying the processing method.
        """
        # Create the gain function
        self.gain_function = np.arange(np.finfo(np.float32).eps, self.t, self.dt)
        
        if len(self.gain_function) != self.nt:
            self.gain_function = np.linspace(np.finfo(np.float32).eps, self.t, self.nt)

        if show_gain:
            # If asked to show the data
            self._show_gain(G.gain_function)
        
    def constant(self, res, s=None):
        """
        Apply a constant gain.

        Parameters
        ----------
        res : dict
            Raw data
        s : ndarray
            Gain value

        Returns
        -------
        dict
            Reformatted seismic section 
        """
        if s is None:
            self.gain_function = np.array([self.dt])
        else:
            self.gain_function = np.array([s])
            
        
    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, value):
        self._test = value

    def _show_gain(self, gain_function):
        """
        show_gain plots the gain function.

        This function plots the gain function

        Args:
            gain_function (float32): The gain function.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(gain_function, np.arange(self.nt))
        ax.invert_yaxis()
        ax.grid()
        ax.set_xlabel("Gain amplitude")
        ax.set_ylabel("Time sample")

class GainFunction(Gain):
    def __init__(self, t, dt, nt):
        """
        This class works with the choosen gain

        This class gets the gain function and apply it

        Args:
            GainFunction (float32): The gain function
        """
        Gain.__init__(self, t, dt, nt)

    def apply(self, data,  show=False):
        """
        apply applies the gain function on the data

        apply applies the made gain on the data. It can show the original and gained data as well.

        Args:
            data (list): A list containing the seismic datafor different component
            show (bool, optional): The option to let user to ask to show the seismic sections before and after gain. Defaults to False.

        Returns:
            gained_data [list]: Gained data
        """
        self.data = data
        gain = self.gain_function.reshape(-1,1)
        
        if type(data).__name__ == 'list':
            self.gain_2d = np.dot(gain, np.ones((1, data[0].shape[1]), np.float32))
            self.gained_data = [self.gain_2d * comp_i for comp_i in data]
            
        elif type(data).__name__ == 'ndarray':
            self.gain_2d = np.dot(gain, np.ones((1, data.shape[1]), np.float32))
            self.gained_data = self.gain_2d *  data

        
        if show:
            fig = plt.figure()

            ax = fig.add_subplot(1, 2, 1)
            seismic_section(ax, data)

            ax = fig.add_subplot(1, 2, 2)
            seismic_section(ax, self.gained_data)
            ax.set_yticks([])

        return self.gained_data
    
    
def prepare_residual(res, s=1.):
    """
    prepare_residual prepares the seismic data as the desire format of FWI class.

    Parameters
    ----------
    res : dict
        Seismic section
    s : ndarray
        Parqameter to create the square matirix of W as the weight of seismic data in cost function.

    Returns
    -------
    dict
        Reformatted seismic section 
    """
    
    data = {}
    shape = res[[*res][0]].shape
    all_comps = ['vx', 'vz', 'taux', 'tauz', 'tauxz']
    
    for param in all_comps:
        if param in res:
            data[param] = s * res[param]
        else:
            data[param] = np.zeros(shape, np.float32)
    return data

if __name__ == "__main__":
    import PyFWI.seismic_io as io

    data = io.load_mat('/Users/amir/repos/seismic/src/PyFWI/data/test/bl_data.mat')
    data = data['bl']
    G = GainFunction(t=0.45, dt=0.00061, nt=data.shape[0])
    G.time_linear(False)
    plt.show()
    
    gained = G.apply(data)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    seismic_section(ax, data)
    ax.set_title("Original data")
    
    ax = fig.add_subplot(1, 2, 2)
    seismic_section(ax, gained)
    ax.set_title("Gained data")
    plt.show()
    
    