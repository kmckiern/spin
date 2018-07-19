from __future__ import division
import numpy as np
import copy
from spin.operators import measure_energy


def flip_spin(configuration):
    """ Flip random spin on lattice """

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


def acceptance_criterion(energy_i, energy_f, T):
    """ Gibbs acceptance """

    energy_difference = 2 * (energy_f - energy_i)
    gibbs_criterion = np.exp(-1. * energy_difference / T)
    if (energy_difference < 0) or (np.random.rand() < gibbs_criterion):
        return True


def mc_step(J, T, energy, configuration):
    """ To take a step, flip a spin and check for acceptance """

    while True:
        # flip spin
        configuration_n = flip_spin(configuration)
        energy_n = measure_energy(J, configuration_n)
        # accept according to acceptance criterion
        if acceptance_criterion(energy, energy_n, T):
            return configuration_n, energy_n


def check_convergence(energies, threshold=.01):
    """ Converged if standard error of the energy < threshold """

    ste = np.std(energies) / (len(energies) ** .5)
    if ste < threshold:
        return True


def check_autocorrelation(configurations, energies, desired_samples, threshold=.01, min_lag=50):
    """ Determine autocorrelation of time series """

    energies -= np.mean(energies)
    n_samples = len(energies)
    for lag in np.arange(min_lag, n_samples, 2):
        ac = np.corrcoef(energies[:n_samples - lag], energies[lag:n_samples])[0, 1]
        if np.abs(ac) < threshold:
            uncorrelated = configurations[::lag]
            if len(uncorrelated) > desired_samples:
                ensemble = np.array(uncorrelated[:desired_samples])
                ensemble_energies = np.array(energies[::lag][:desired_samples])
                return ensemble, ensemble_energies
            else:
                break
    return lag


def run_mcmc(self, eq=False, min_steps=10000, auto_multiplier=1.4):
    """ Generate samples
        for mixing: until convergence criterion is met
        for eq: until desired number of independent samples found
    """

    configurations = []
    energies = []
    while True:
        self.configuration, self.energy = self.mc_step()
        configurations.append(self.configuration)
        energies.append(self.energy)
        if len(energies) > min_steps:
            if eq:
                if self.check_convergence(energies):
                    break
                else:
                    # if not converged, run for 2x more steps
                    min_steps *= 2
            else:
                autoc = self.check_autocorrelation(configurations, energies)
                if autoc == True:
                    break
                else:
                    # if not enough configurations, run for lag more steps
                    min_steps = self.n_samples * autoc * auto_multiplier


def sample(self):
    """ Run MCMC scheme on initial configuration """

    # equilibrate (mix) the chain
    self.run_mcmc(eq=True)

    # production, get at least n_samples samples
    self.run_mcmc()
