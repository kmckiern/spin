import numpy as np
from scipy.ndimage import filters 

class Operators(object):

    """
    Measure properties of particle system configuration
    """

    def __init__(self, J=-1.0):
        if self.configuration.any() == None:
            raise ValueError('must have a configuration!')
        self.energy = self.measure_energy(self.configuration, J)
        self.magnetization = self.measure_magnetization(self.configuration)

    def adj_kernel(self, configuration):

        """
        Creates adjecency kernel for arbitrary dimensional array
        """

        # create kernel of correct shape
        kernel = np.ones(configuration.shape)
        kernel = kernel[tuple(slice(0, 3) for i in kernel.shape)]
        # zero non adjacent / center
        non_adj = kernel[tuple(slice(None, None, j-1) for j in kernel.shape)]
        non_adj *= 0
        center = kernel[tuple(slice(1, 2, j) for j in kernel.shape)]
        center *= 0
        return kernel

    def hamiltonian(self, configuration, J):

        """
        Evaluate hamiltonian via normalized convolution with adjacency kernel
        """

        kernel = self.adj_kernel(configuration) 
        c = filters.convolve(configuration, kernel, mode='wrap')
        return J * np.sum(c * configuration) / np.sum(kernel)
    
    def measure_energy(self, configuration, J):
        
        """
        Calculate energy for arbitrary dimensional configuration
        """
        
        if configuration.ndim > 2:
            return np.array([self.hamiltonian(c, J=J) for c in configuration])
        else:
            return self.hamiltonian(configuration, J=J)
    
    def measure_magnetization(self, configuration):

        """
        Given by total spin value
        """

        if configuration.ndim > 2:
            n_spin = configuration[0].size
        else:
            n_spin = configuration.size

        return configuration.sum(-1).sum(-1) / n_spin
    
