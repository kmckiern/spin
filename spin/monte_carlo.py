import numpy as np
from .operators import *
import matplotlib.pyplot as plt

"""
Sample system via MCMC with Gibbs Sampling
"""

def flip_spin(configuration, geometry):
    """
    Flip random spin on lattice
    """
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

def check_convergence(energies, threshold=1):
    """
    Converged if standard error of the energy < threshold
    """
    energies = energies[10:]
    ste = np.std(energies) / (len(energies)**.5)
    if ste < threshold:
        return False
    else:
        return True

def check_autocorrelation(energies):
    """
    Determine autocorrelation of time series
    """
    energies -= np.mean(energies)
    corr = np.correlate(energies, energies, mode='full')
    n_samp = len(energies)
    var = np.var(energies)
    ac = corr / var
    ac_l = int(ac.argmax() * .5)
    return ac_l

def run_MCMC(configuration, geometry, energy, T, n_samples=1, min_samples=10000):
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
                if n_samples == 1:
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
                    print (ac)
                    if len(configurations) > ac*n_samples:
                        continue
                    else:
                        configs = configurations[::ac]                    
                        return configs

def sample(config, geo, energy, T, n_samples=3):
    """
    Run MCMC scheme on initial configuration
    """
    # equilibrate (mix) the chain
    config, energy = run_MCMC(config, geo, energy, T)

    # get n_samples samples
    configurations = run_MCMC(config, geo, energy, T, n_samples)

    return configurations
