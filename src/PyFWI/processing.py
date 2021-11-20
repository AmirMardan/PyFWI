import numpy as np
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
import PyFWI.seismic_io as io

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

class Gain_function(Gain):
    def __init__(self, t, dt, nt):
        """
        This class works with the choosen gain

        This class gets the gain function and apply it

        Args:
            gain_function (float32): The gain function
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
            self.gain_2d = gain @ np.ones((1, data[0].shape[1]), np.float32)
            self.gained_data = [self.gain_2d * comp_i for comp_i in data]
            
        elif type(data).__name__ == 'ndarray':
            self.gain_2d = gain @ np.ones((1, data.shape[1]), np.float32)
            self.gained_data = self.gain_2d *  data

        
        if show:
            fig = plt.figure()

            ax = fig.add_subplot(1, 2, 1)
            seismic_section(ax, data)

            ax = fig.add_subplot(1, 2, 2)
            seismic_section(ax, self.gained_data)
            ax.set_yticks([])

        return self.gained_data


class derivatives(object):
    def __init__(self, order):
        """
        derivatives is a class to implenet the the derivatives for wave modeling

        The coeeficients are based on Lavendar, 1988 and Hasym et al., 2014.

        Args:
            order (int, optional): [description]. Defaults to 4.
        """
        self._order = order
        
        if order == 4:
            self._c1 = 9/8
            self._c2 = - 1 / 24
            self._c3 = 0
            self._c4 = 0

        elif order == 8:
            self._c1 = 1715 / 1434
            self._c2 = -114 / 1434
            self._c3 = 14 / 1434
            self._c4 = -1 / 1434
            
        else:
            raise AssertionError ("Order of the derivative has be either 4 or 8!")

        dh_n = { # Dablain, 1986, Bai et al., 2013
            '4': 4,
            '8': 3
        } 
        
        self.dh_n = dh_n[str(order)] # Pick the appropriate n for calculating dh
        
    @property
    def order(self):
        return self._order
    
    @order.setter
    def order(self, value):
        if value in [4, 8]:
            self._order = value
        else:
            raise AssertionError ("Order of the derivative has be either 4 or 8!")
    
    @property
    def c1(self):
        return self._c1
        
    @c1.setter
    def c1(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")
    
    @property
    def c2(self):
        return self._c2
        
    @c2.setter
    def c2(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")
    
    @property
    def c3(self):
        return self._c3
        
    @c3.setter
    def c3(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")
    
    @property
    def c4(self):
        return self._c4
        
    @c4.setter
    def c3(self, value):
        raise AttributeError("Denied! You can't change the derivative's coefficients")
    
    
    def dxp(self, x, dx):
         if self.order == 4:
             return self._dxp4(x, dx)
         else:
             return self._dxp8(x, dx)
         
    def dxn(self, x, dx):
         if self.order == 4:
             return self._dxn4(x, dx)
         else:
             return self._dxn8(x, dx)
         
    def dzp(self, x, dx):
         if self.order == 4:
             return self._dzp4(x, dx)
         else:
             return self._dzp8(x, dx)
         
    def dzn(self, x, dx):
         if self.order == 4:
             return self._dzn4(x, dx)
         else:
             return self._dzn8(x, dx)
         
         
        
    def _dxp4(self, x, dx):
        y = np.zeros(x.shape)
        
        y[2:-2, 2:-2] = (self._c1 * (x[2:-2, 3:-1] - x[2:-2, 2:-2]) +
                         self._c2 * (x[2:-2, 4:] - x[2:-2, 1:-3])) / dx
        return y
    
    def _dxp8(self, x, dx): 
        y = np.zeros(x.shape)
        
        y[4:-4, 4:-4] = (self._c1 * (x[4:-4, 5:-3] - x[4:-4, 4:-4]) +
                         self._c2 * (x[4:-4, 6:-2] - x[4:-4, 3:-5]) + 
                         self._c3 * (x[4:-4, 7:-1] - x[4:-4, 2:-6]) + 
                         self._c4 * (x[4:-4, 8:] - x[4:-4, 1:-7])) / dx
        return y


    def _dxn4(self, x, dx):        
        y = np.zeros((x.shape))
        
        y[2:-2, 2:-2] = (self._c1 * (x[2:-2, 2:-2] - x[2:-2, 1:-3]) + 
                         self._c2 * (x[2:-2, 3:-1] - x[2:-2, :-4])) / dx
        return y 
    
    def _dxn8(self, x, dx):
        y = np.zeros((x.shape))
        
        y[4:-4, 4:-4] = (self._c1 * (x[4:-4, 4:-4] - x[4:-4, 3:-5]) +
                         self._c2 * (x[4:-4, 5:-3] - x[4:-4, 2:-6]) +
                         self._c3 * (x[4:-4, 6:-2] - x[4:-4, 1:-7]) +
                         self._c4 * (x[4:-4, 7:-1] - x[4:-4, :-8]) ) / dx
        return y 

    def _dzp4(self, x, dx):
        y = np.zeros(x.shape)
        
        y[2:-2, 2:-2] = (self.c1 * (x[3:-1, 2:-2] - x[2:-2, 2:-2]) +
                         self.c2 * (x[4:, 2:-2] - x[1:-3, 2:-2])) / dx
        return y
    
    def _dzp8(self, x, dx):
        y = np.zeros(x.shape)
        
        y[4:-4, 4:-4] = (self.c1 * (x[5:-3, 4:-4] - x[4:-4, 4:-4]) +
                         self.c2 * (x[6:-2, 4:-4] - x[3:-5, 4:-4]) +
                         self.c2 * (x[7:-1, 4:-4] - x[2:-6, 4:-4]) +
                         self.c2 * (x[8:, 4:-4] - x[1:-7, 4:-4])) / dx
        return y


    def _dzn4(self, x, dx):
        y = np.zeros((x.shape))
        
        y[2:-2, 2:-2] = (self.c1 * (x[2:-2, 2:-2] - x[1:-3, 2:-2]) +
                         self.c2 * (x[3:-1, 2:-2] - x[:-4, 2:-2])) / dx
        return y 
    
    def _dzn8(self, x, dx):
        y = np.zeros((x.shape))
        
        y[4:-4, 4:-4] = (self.c1 * (x[4:-4, 4:-4] - x[3:-5, 4:-4]) +
                         self.c2 * (x[5:-3, 4:-4] - x[2:-6, 4:-4]) +
                         self.c2 * (x[6:-2, 4:-4] - x[1:-7, 4:-4]) +
                         self.c2 * (x[7:-1, 4:-4] - x[:-8, 4:-4])) / dx
        return y 


    def dot_product_test_derivatives(self):
        
        x = np.random.rand(100, 100)
        x[:4, :] = x[-4:, :] = x[:, :4] = x[:, -4:] = 0

        y = np.random.rand(100, 100)
        y[:4, :] = y[-4:, :] = y[:, :4] = y[:, -4:] = 0

        error_x = np.sum(x * self.dxp(y, 1)) - np.sum(- self.dxn(x, 1) * y)
        error_z = np.sum(x * self.dzp(y, 1)) - np.sum(- self.dzn(x, 1) * y)

        print(f"Errors for derivatives are \n {error_x = }, {error_z = }")    

    def dt_computation(self, vp_max, dx, dz=None):
        '''
        ref: Bai et al, 2013
        '''
        if dz is None:
            dz = dx
        
        c_sum = np.abs(self._c1) + np.abs(self._c2) + \
            np.abs(self._c3) + np.abs(self._c4)
        
        a = 1/dx/dx * c_sum + 1/dz/dz * c_sum
        dt = 2 / vp_max / np.sqrt(a*(1 + 4.0)) 
        
        return dt
        
    
if __name__ == "__main__":
    data = io.load_mat('/Users/amir/repos/seismic/src/PyFWI/data/test/bl_data.mat')
    data = data['bl']
    G = Gain_function(t=0.45, dt=0.00061, nt=data.shape[0])
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
    
    