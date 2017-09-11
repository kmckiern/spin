import numpy as np
from .operators import conf_energy, magnetization
from .monte_carlo import sample

class System(object):
    """
    Representation of a particle system

    Parameters
    ----------
    spin : spin value for particles
    geometry : geometric arrangement of particles
    configuration : spin configuration of particles
    T: temperature
    """
    def __init__(self, spin=1, geometry=(1,), configuration=None, T=1):
        """
        Returns particle system object 
        """
        if isinstance(geometry, int):
            geometry = (geometry,)

        self._spin = spin
        self._geometry = geometry
        self._configuration = configuration
        self._T = T

    def random_configuration(self):
        """
        Distribute binary particles according to random configuration
        """
        state = np.random.choice([-1, 1], size=self._geometry)
        state *= self._spin
        self._configuration = state

    def uniform_configuration(self, val):
        """
        Distribute particles according to uniform configuration
        """
        state = np.ones(self._geometry) * val
        self._configuration = state

    def energy(self):
        """
        Calculate configuration energy
        """
        return conf_energy(self._configuration)

    def mag(self):
        """
        Calculate configuration magnetization
        """
        return magnetization(self._configuration)

    def sample_MCMC(self):
        """
        Equilibrate via Metroplis Monte Carlo
        """
        samples = sample(self._configuration, self._geometry, self.energy(), self._T)
        return samples
