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
    accept = False
    energy_difference = energy_j - energy_i
    if energy_difference < 0:
        accept = True
    else:
        if np.random.random() < np.exp(-1. * energy_difference / T):
            accept = True
    return accept

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
        return None

def check_convergence(energies, threshold=.3):
    """
    Converged if standard error of the energy < threshold
    """
    ste = np.std(energies) / (len(energies)**.5)
    if ste < threshold:
        return False
    else:
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

def run_MCMC(configuration, geometry, energy, T, eq=False, min_samples=10000):
    """
    Generate samples
        for mixing: until convergence criterion is met
        for eq: until desired number of independent samples found
    """
    configurations = []
    energies = []
    continue_sampling = True
    while continue_sampling:
        # generate sample
        sample = MC_step(configuration, geometry, energy, T)
        if sample == None:
            continue
        else:
            configuration, energy = sample
            configurations.append(configuration)
            energies.append(energy)
            if len(energies) > min_samples:
                if eq:
                    # check convergence
                    continue_sampling = check_convergence(energies)
                    # if not converged, run for 10% more steps
                    if continue_sampling:
                        min_samples *= .1
                    else:
                        return configuration, energy
                else:
                    # check autocorrelation
                    ac = check_autocorrelation(energies)
                    if ac == 0 or (len(configurations) < ac*n_samples):
                        min_samples *= .1
                        continue
                    else:
                        return configurations[::ac]

def sample(config, geo, energy, T, n_samples=100):
    """
    Run MCMC scheme on initial configuration
    """
    # equilibrate (mix) the chain
    config, energy = run_MCMC(config, geo, energy, T, eq=True, min_samples=50000)

    # get n_samples samples
    configurations = run_MCMC(config, geo, energy, T)

    # return ensemble of samples
    return np.array(configurations)

