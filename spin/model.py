import numpy as np
from ensemble import run_mcmc

class Model:
    """ Create, equilibrate, measure, and build network of model """

    def __init__(self,
                 J=1,
                 T=1,
                 geometry=(1,)):
        self.J = J
        self.T = T
        self.geometry = geometry

    def random_configuration(self):
        """ Distribute particles according to random configuration """
        configuration = np.random.choice([-1, 1], size=self.geometry)
        self.configuration = configuration

    def uniform_configuration(self, val=1):
        """ Distribute particles according to uniform configuration """
        configuration = np.ones(self.geometry)
        self.configuration = configuration

    def generate_ensemble(self, ensemble_size, eq=True):
        if eq:
            self.configuration = equilibrate_configuration(self.J, self.T, self.configuration)
        self.ensemble, self.energies = run_mcmc(self.J, self.T, self.configuration, ensemble_size)


def correct_hyper_dict(hypers, optimize):
    for element in hypers.keys():
        val = hypers[element]
        if optimize:
            if not isinstance(val, list):
                hypers[element] = [val]
        else:
            if isinstance(val, list):
                hypers[element] = val[0]
    return hypers
