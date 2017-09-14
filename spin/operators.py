import numpy as np
from scipy.ndimage import filters 

class Operators(object):

    """
    Measure properties of particle system configuration
    """

    def __init__(self, configuration, J=-1.0):
        if configuration == None:
            raise ValueError('must have a configuration!')
        self._energy = self.energy(configuration, J)
        self._magnetization = self.magnetization(configuration)
    
    def energy(self, configuration, J):

        """
        Evaluate hamiltonian via normalized convolution with adjacency kernel
        """

        # create kernel of correct shape
        kernel = np.ones(configuration.shape)
        kernel = kernel[tuple(slice(0, 3) for i in kernel.shape)]
        # zero non adjacent / center
        non_adj = kernel[tuple(slice(None, None, j-1) for j in kernel.shape)]
        non_adj *= 0
        center = kernel[tuple(slice(1, 2, j) for j in kernel.shape)]
        center *= 0

        c = filters.convolve(configuration, kernel, mode='wrap')
        return J * np.sum(c * configuration) / np.sum(kernel)
    
    def magnetization(self, configuration):

        """
        Given by total spin value
        """

        return np.sum(configuration) / configuration.size
    
