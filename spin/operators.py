from __future__ import division
import numpy as np
from scipy.ndimage import filters 


class Operators(object):

    """ Measure properties of particle system configuration """

    def __init__(self, J=1.0):
        if self.configuration.any() == None:
            raise ValueError('must have a configuration!')
        self.J = J
        self.energy = self.measure_energy(self.configuration)
        self.magnetization = self.measure_magnetization(self.configuration)

    def adj_kernel(self, configuration):

        """ Creates adjecency kernel for arbitrary dimensional array """

        # create kernel of correct shape
        kernel = np.ones(configuration.shape)
        kernel = kernel[tuple(slice(0, 3) for i in kernel.shape)]

        # zero non adjacent / center
        if kernel.ndim > 1:
            non_adj = kernel[tuple(slice(None, None, j-1) for j in kernel.shape)]
            non_adj *= 0
        center = kernel[tuple(slice(j-2, j-1, j) for j in kernel.shape)]
        center *= 0
        return kernel

    def hamiltonian(self, configuration):

        """ Evaluate hamiltonian via normalized convolution with kernel """

        kernel = self.adj_kernel(configuration) 
        c = filters.convolve(configuration, kernel, mode='wrap')
        return -1. * self.J * np.sum(c * configuration) / np.sum(kernel)
    
    def measure_energy(self, configuration):
        
        """ Calculate energy for arbitrary dimensional configuration """

        if configuration.ndim > 2:
            return np.array([self.hamiltonian(c) for c in configuration])
        else:
            return self.hamiltonian(configuration)
    
    def measure_magnetization(self, configuration):

        """ Given by normalized sum over all spin values """

        n_spin = np.prod(self.geometry)

        for d in range(len(self.geometry)):
            if d == 0:
                mag = configuration.sum(-1)
            else:
                mag = mag.sum(-1)
        mag = mag / n_spin
        mag = np.abs(mag)

        return mag
