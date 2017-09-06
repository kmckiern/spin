import numpy as np
from .operators import *

class system(object):
    """
    Representation of a particle system

    Parameters
    ----------
    spin : spin value for particles
    geometry : geometric arrangement of particles
    configuration : spin configuration of particles
    """

    def __init__(self, spin=.5, geometry=(1,), configuration=None):
        """
        Returns particle system object 
        """
        if isinstance(geometry, int):
            geometry = (geometry,)

        self._spin = spin
        self._geometry = geometry
        self._configuration = configuration

    def randomly_distributed(self):
        """
        Distribute particles according to random configuration
        """
        state = np.random.randint(2, size=self._geometry) - self._spin
        self._configuration = state

    def uniformly_distributed(self, val):
        """
        Distribute particles according to uniform configuration
        """
        state = np.ones(self._geometry) * val
        self._configuration = state

    def mag(self):
        """
        Calculate configuration magnetization
        """
        return magnetization(self._configuration)
