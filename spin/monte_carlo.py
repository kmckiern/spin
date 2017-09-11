import numpy as np
from scipy.signal import correlate
from .operators import *
import copy

"""
Sample system via MCMC with Gibbs Sampling
"""

def flip_spin(configuration, geometry):
    """
    Flip random spin on lattice
    """
    configuration = copy.copy(configuration)
    # identify random index
    flip_indices = []
    for dimension in geometry:
        if isinstance(dimension, int):
            index = np.random.randint(dimension)
            flip_indices.append(index)
    flip_indices = tuple(flip_indices)
    # flip
    configuration[flip_indices] *= -1
    return configuration

def acceptance_criterion(energy_i, energy_j, T):
    """
    Gibbs acceptance
    """
    energy_difference = energy_j - energy_i
    gibbs_criterion = np.exp(-1. * energy_difference / T)
    if (energy_difference < 0) or (np.random.rand() < gibbs_criterion):
        return True

def MC_step(configuration, geometry, energy, T):
    """
    To take a step, flip a spin and check for acceptance
    """
    # flip spin
    configuration_n = flip_spin(configuration, geometry)
    energy_n = conf_energy(configuration_n)
    # accept according to acceptance criterion
    if acceptance_criterion(energy, energy_n, T):
        return configuration_n, energy_n
    else:
        return MC_step(configuration, geometry, energy, T)

def check_convergence(energies, threshold=.3):
    """
    Converged if standard error of the energy < threshold
    """
    ste = np.std(energies) / (len(energies)**.5)
    if ste > threshold:
        return True

def check_autocorrelation(energies, threshold=.01):
    """
    Determine autocorrelation of time series
    """
    energies -= np.mean(energies)
    n_samples = len(energies)
    for lag in np.arange(0, n_samples, int(.05*n_samples)):
        ac = np.corrcoef(energies[:n_samples-lag], energies[lag:n_samples])[0,1]
        if ac < threshold:
            return lag
    return 0

def run_MCMC(configuration, geometry, energy, T, n_samples, eq=False, min_steps=10000):
    """
    Generate samples
        for mixing: until convergence criterion is met
        for eq: until desired number of independent samples found
    """
    continue_sampling = True
    configurations = []
    energies = []
    while continue_sampling:
        configuration, energy = MC_step(configuration, geometry, energy, T)
        configurations.append(configuration)
        energies.append(energy)
        if len(energies) > min_steps:
            if eq:
                continue_sampling = check_convergence(energies)
                if not continue_sampling:
                    return configuration, energy
            else:
                ac = check_autocorrelation(energies)
                if ac != 0 and (len(configurations) > ac*n_samples):
                    return configurations[::ac]
            # if not converged, or too correlated, run for 10% more steps
            min_steps *= .1

def sample(config, geo, energy, T, n_samples=3):
    """
    Run MCMC scheme on initial configuration
    """
    # equilibrate (mix) the chain
    config, energy = run_MCMC(config, geo, energy, T, 1, eq=True, min_steps=50000)

    # get at least n_samples samples
    configurations = run_MCMC(config, geo, energy, T, n_samples)

    # return ensemble of samples
    return np.array(configurations)

