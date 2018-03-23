import numpy as np
from scipy.ndimage import filters 
from spin.operators import Operators

class System(Operators):

    """ Representation of a binary particle system

    Parameters
    ----------
    T: temperature
    spin : spin value for particles
    geometry : geometric arrangement of particles
    configuration : spin configuration of particles
    """

    def __init__(self, T=1, spin=1, geometry=(1,), configuration=None):

        """ Returns particle system object """

        self.T = T
        self.spin = spin
        if isinstance(geometry, int):
            geometry = (geometry,)
        if geometry[-1] == 1:
            geometry = (geometry[0],)
        self.geometry = geometry
        if configuration is None:
            self.configuration = self.random_configuration()
        else:
            self.configuration = configuration

        # measure system
        super(System, self).__init__()

    def random_configuration(self):

        """ Distribute particles according to random configuration """

        state = np.random.choice([-1, 1], size=self.geometry)
        state *= self.spin
        return state

    def uniform_configuration(self, val=1):

        """ Distribute particles according to uniform configuration """

        state = np.ones(self.geometry) * val
        return state
