import numpy as np
from scipy.ndimage import filters 

"""
Measure properties of particle system configuration
"""

def adj_kernel(configuration):
    """
    Adjacency filter matrix of arbitrary dimension
    """
    # create kernel of correct shape
    kernel = np.ones(configuration.shape)
    kernel = kernel[tuple(slice(0, 3) for i in kernel.shape)]
    # zero non adjacent 
    non_adj = kernel[tuple(slice(None, None, j-1) for j in kernel.shape)]
    non_adj *= 0
    # zero center
    center = kernel[tuple(slice(1, 2, j) for j in kernel.shape)]
    center *= 0
    return kernel

def conf_energy(configuration, J=-1.0):
    """
    Evaluate hamiltonian via normalized convolution with adjacency kernel
    """
    kernel = adj_kernel(configuration)
    n_adj = np.sum(kernel)
    c = filters.convolve(configuration, kernel, mode='wrap')
    energy = J * np.sum(c * configuration) / n_adj 
    return energy

def magnetization(configuration):
    """
    Given by total spin value
    """
    return np.sum(configuration) / configuration.size

