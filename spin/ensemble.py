from __future__ import division
import numpy as np
import copy
from spin.operators import measure_energy


def flip_spin(configuration, seed=None):
    """ Flip random spin on lattice """

    if seed is not None:
        np.random.seed(seed)

    # identify random index
    flip_indices = []
    for dimension in configuration.shape:
        if isinstance(dimension, int):
            index = np.random.randint(dimension)
            flip_indices.append(index)
    flip_indices = tuple(flip_indices)

    configuration = copy.copy(configuration)
    configuration[flip_indices] *= -1
    return configuration


def acceptance_criterion(energy_i, energy_f, T, seed=None):
    """ Gibbs acceptance """

    if seed is not None:
        np.random.seed(seed)

    energy_difference = 2 * (energy_f - energy_i)
    gibbs_criterion = np.exp(-1. * energy_difference / T)
    if (energy_difference < 0) or (np.random.rand() < gibbs_criterion):
        return True


def mc_step(J, T, configuration, seed=None):
    """ To take a step, flip a spin and check for acceptance """

    if seed is not None:
        np.random.seed(seed)

    energy = measure_energy(J, configuration)

    while True:
        # flip spin
        configuration_n = flip_spin(configuration, seed)
        energy_n = measure_energy(J, configuration_n)

        # accept according to acceptance criterion
        if acceptance_criterion(energy, energy_n, T, seed):
            return configuration_n, energy_n
        else:
            if seed is not None:
                break


def check_convergence(energies, threshold=.1):
    """ Converged if standard error of the energy < threshold """

    ste = np.std(energies) / (len(energies) ** .5)
    if ste < threshold:
        return True


def check_autocorrelation(configurations, energies, desired_samples, threshold=.05, min_lag=10):
    """ Determine autocorrelation of time series """

    energies -= np.mean(energies)
    n_samples = len(energies)
    for lag in np.arange(min_lag, n_samples, 2):
        ac = np.corrcoef(energies[:n_samples - lag], energies[lag:n_samples])[0, 1]

        if np.abs(ac) < threshold:
            uncorrelated = configurations[::lag]
            if len(uncorrelated) >= desired_samples:
                return lag
            else:
                return np.inf
    return np.inf


def run_mcmc(J, T, configuration, desired_samples=1, min_step_multiplier=100, autocorrelation_threshold=.1, seed=None):
    """ Generate samples until either:
        - convergence criterion is met (chain is `mixed`)
        - desired number of independent samples found
    """
    min_steps = configuration.size * min_step_multiplier

    if seed is not None:
        np.random.seed(seed)

    configurations = []
    energies = []

    while True:
        configuration, energy = mc_step(J, T, configuration, seed)
        configurations.append(configuration)
        energies.append(energy)

        if len(energies) > min_steps:

            if desired_samples == 1:
                if check_convergence(energies):
                    return configurations[-1]

            else:
                lag = check_autocorrelation(configurations, energies, desired_samples,
                                            threshold=autocorrelation_threshold)

                if lag < np.inf:
                    ensemble = np.array(configurations)[::lag][:desired_samples]
                    energies = np.array(energies)[::lag][:desired_samples]
                    return ensemble, energies

            min_steps *= 2
