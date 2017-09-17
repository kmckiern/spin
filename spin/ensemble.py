import numpy as np
import copy
from .operators import Operators

class Ensemble(Operators):

    """
    Sample system via MCMC with Gibbs Sampling
    """

    def __init__(self, system, n_samples=1):

        self.__dict__.update(system.__dict__)
        self.n_samples = n_samples
        self.configuration = self.sample()

        # measure ensemble
        super(Ensemble, self).__init__()
    
    def flip_spin(self):

        """
        Flip random spin on lattice
        """

        # identify random index
        flip_indices = []
        for dimension in self.geometry:
            if isinstance(dimension, int):
                index = np.random.randint(dimension)
                flip_indices.append(index)
        flip_indices = tuple(flip_indices)

        configuration = copy.copy(self.configuration)
        configuration[flip_indices] *= -1
        return configuration
    
    def acceptance_criterion(self, energy_f):

        """
        Gibbs acceptance
        """

        energy_difference = energy_f - self.energy
        gibbs_criterion = np.exp(-1. * energy_difference / self.T)
        if (energy_difference < 0) or (np.random.rand() < gibbs_criterion):
            return True
    
    def mc_step(self):

        """
        To take a step, flip a spin and check for acceptance
        """

        while True:
            # flip spin
            configuration_n = self.flip_spin()
            energy_n = self.measure_energy(configuration_n, J=-1.0)
            # accept according to acceptance criterion
            if self.acceptance_criterion(energy_n):
                return configuration_n, energy_n
    
    def check_convergence(self, energies, threshold=.3):

        """
        Converged if standard error of the energy < threshold
        """

        ste = np.std(energies) / (len(energies)**.5)
        if ste < threshold:
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
    
    def run_mcmc(self, eq=False, min_steps=100):

        """
        Generate samples
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
                    acl = self.check_autocorrelation(energies)
                    if acl != None:
                        id_configurations = configurations[::acl]
                        if len(id_configurations) > self.n_samples:
                            return id_configurations[:self.n_samples]
                # if not converged/too correlated, run for 2x more steps
                min_steps *= 2

    def sample(self):

        """
        Run MCMC scheme on initial configuration
        """
        
        # equilibrate (mix) the chain
        self.run_mcmc(eq=True)
    
        # production, get at least n_samples samples
        configurations = self.run_mcmc()
    
        return np.array(configurations)
