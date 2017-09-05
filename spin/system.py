import numpy as np

class system(object):
    """
    Representation of a particle system

    Parameters
    ----------
    spin : spin value for particles
    geometry : geometric arrangement of particles
    distribution : spin distribution of particle system
    """

    def __init__(self, spin=.5, geometry=(1,), distribution=None):
        """
        Returns a system object with a given spin distribution
        """

        if isinstance(geometry, int):
            geometry = (geometry,)

        self._spin = spin
        self._geometry = geometry
        self._distribution = distribution

    def randomly_distributed(self):
        """
        Distribute particles according to random distribution
        """
        state = np.random.randint(2, size=self._geometry) * self._spin
        self._distribution = state

    def uniformly_distributed(self, val):
        """
        Distribute particles according to uniform distribution
        """
        state = np.ones(self._geometry) * val
        self._distribution = state
