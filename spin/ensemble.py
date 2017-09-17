import numpy as np
import copy
from .operators import Operators

class Ensemble(Operators):

    """
    Sample system via MCMC with Gibbs Sampling
    """

    def __init__(self, system, n_samples):

        self.configuration = self.sample(system.configuration,
                system.geometry, system.energy, system.T,
                n_samples)

        # measure ensemble
        super(Ensemble, self).__init__()
    
    def flip_spin(self, configuration, geometry):

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

        configuration = copy.copy(configuration)
        configuration[flip_indices] *= -1
        return configuration
    
    def acceptance_criterion(self, energy_i, energy_f, T):

        """
        Gibbs acceptance
        """

        energy_difference = energy_f - energy_i
        gibbs_criterion = np.exp(-1. * energy_difference / T)
        if (energy_difference < 0) or (np.random.rand() < gibbs_criterion):
            return True
    
    def MC_step(self, configuration, geometry, energy, T):

        """
        To take a step, flip a spin and check for acceptance
        """

        while True:
            # flip spin
            configuration_n = self.flip_spin(configuration, geometry)
            energy_n = self.energy(configuration_n, J=-1.0)
            # accept according to acceptance criterion
            if self.acceptance_criterion(energy, energy_n, T):
                return configuration_n, energy_n
    
    def check_convergence(self, energies, threshold=.3):

        """
        Converged if standard error of the energy < threshold
        """

        ste = np.std(energies) / (len(energies)**.5)
        if ste > threshold:
            return True
    
    def check_autocorrelation(self, energies, threshold=.01):

        """
        Determine autocorrelation of time series
        """

        energies -= np.mean(energies)
        n_samples = len(energies)
        for lag in np.arange(0, n_samples, 2):
            ac = np.corrcoef(energies[:n_samples-lag], 
                    energies[lag:n_samples])[0,1]
            if ac < threshold:
                return lag
    
    def run_MCMC(self, configuration, geometry, energy, T, n_samples=1,
            eq=False, min_steps=100):

        """
        Generate samples
            for mixing: until convergence criterion is met
            for eq: until desired number of independent samples found
        """

        configurations = []
        energies = []
        while True:
            configuration, energy = self.MC_step(configuration, geometry,
                    energy, T)
            configurations.append(configuration)
            energies.append(energy)
            if len(energies) > min_steps:
                if eq:
                    if not self.check_convergence(energies):
                        return configuration, energy
                else:
                    acl = self.check_autocorrelation(energies)
                    if acl != None:
                        id_configurations = configurations[::acl]
                        if len(id_configurations) > n_samples:
                            return id_configurations[:n_samples]
                # if not converged/too correlated, run for 2x more steps
                min_steps *= 2

    def equilibrate(self, configuration, geometry, energy, T, n_samples):

        """
        Equilibrate (mix) the chain
        """

        return self.run_MCMC(configuration, geometry, energy, T, eq=True)

    
    def sample(self, configuration, geometry, energy, T, n_samples):

        """
        Run MCMC scheme on initial configuration
        """
        
        # equilibration
        eq_configuration, eq_energy = self.equilibrate(configuration, geometry,
                energy, T, n_samples)
    
        # production, get at least n_samples samples
        configurations = self.run_MCMC(eq_configuration, geometry, eq_energy,  
                T, n_samples=n_samples)
    
        return np.array(configurations)
