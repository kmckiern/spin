from __future__ import division
import numpy as np
from scipy.ndimage import filters


def adj_kernel(configuration):
    """ Creates adjecency kernel for arbitrary dimensional array """
    # ensure each dimension is gt 2
    for dim_length in configuration.shape:
        assert dim_length > 2

    # create kernel of correct shape
    kernel = np.ones(configuration.shape)
    kernel = kernel[tuple(slice(0, 3) for i in kernel.shape)]

    # zero non adjacent / center
    if kernel.ndim > 1:
        non_adj = kernel[tuple(slice(None, None, j - 1) for j in kernel.shape)]
        non_adj *= 0
    center = kernel[tuple(slice(j - 2, j - 1, j) for j in kernel.shape)]
    center *= 0
    return kernel


def measure_energy(J, configuration):
    """ Evaluate hamiltonian via normalized convolution with kernel """
    kernel = adj_kernel(configuration)
    c = filters.convolve(configuration, kernel, mode='wrap')
    energy = -1. * J * np.sum(c * configuration) / np.sum(kernel)
    return energy / configuration.size


def measure_magnetization(configuration):
    """ Given by normalized sum over all spin values """
    for d in range(configuration.ndim):
        if d == 0:
            mag = configuration.sum(-1)
        else:
            mag = mag.sum(-1)
    mag = np.abs(mag)

    return mag / configuration.size


def measure_heat_capacity(energy, temperature, n_spin):
    return (np.mean(energy ** 2) - np.mean(energy) ** 2) / (temperature ** 2 * n_spin)


def measure_magnetic_susceptibility(magnetization, temperature, n_spin):
    return (np.mean(magnetization ** 2) - np.mean(magnetization) ** 2) / (temperature * n_spin)
