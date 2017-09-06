import numpy as numpy
from scipy.ndimage import filters 

"""
Measure properties of particle system configuration
"""

def brute_energy(configuration):
    """
    Convolution over Hamiltonian 
    """
    N, M = configuration.shape
    e = 0
    for i in range(len(configuration)):
        for j in range(len(configuration)):
            s = configuration[i,j]
            nb = configuration[(i + 1) % N, j] + configuration[i, (j + 1) % M] + configuration[(i - 1) % N, j] + configuration[i, (j - 1) % M]
            e += -nb * s 
    return e * .25

def adj_kernel(configuration):
    """
    Adjacency filter matrix of arbitrary dimension
    """
    # create kernel of correct shape
    kernel = numpy.ones(configuration.shape)
    kernel = kernel[tuple(slice(0, 3) for i in kernel.shape)]
    # zero non adjacent 
    non_adj = kernel[tuple(slice(None, None, j-1) for j in kernel.shape)]
    non_adj *= 0
    # zero center
    center = kernel[tuple(slice(1, 2, j) for j in kernel.shape)]
    center *= 0
    return kernel

def conv_energy(configuration):
    """
    Evaluate hamiltonian via convolution with adjacency kernel
    """
    kernel = adj_kernel(configuration)
    n_adj = numpy.sum(kernel)
    c = filters.convolve(configuration, kernel, mode='wrap')
    energy = numpy.sum(c * configuration * configuration) / n_adj 
    return energy

def magnetization(configuration):
    """
    Given by total spin value
    """
    return numpy.sum(configuration)

