import numpy as np
from scipy.ndimage import filters 

class System(object):

    """
    Representation of a binary particle system

    Parameters
    ----------
    T: temperature
    spin : spin value for particles
    geometry : geometric arrangement of particles
    configuration : spin configuration of particles
    """

    def __init__(self, T=1, spin=1, geometry=(1,), configuration=None):

        """
        Returns particle system object 
        """

        self._T = T
        self._spin = spin
        if isinstance(geometry, int):
            geometry = (geometry,)
        self._geometry = geometry
        if configuration == None:
            self._configuration = self.random_configuration()
        else:
            self._configuration = configuration

    def random_configuration(self):

        """
        Distribute particles according to random configuration
        """

        state = np.random.choice([-1, 1], size=self._geometry)
        state *= self._spin
        return state

    def uniform_configuration(self, val):

        """
        Distribute particles according to uniform configuration
        """

        state = np.ones(self._geometry) * val
        return state
