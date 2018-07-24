import os
import pickle

import numpy as np

from spin.ensemble import run_mcmc


class Model:
    """ Create, equilibrate, measure, and build network of model """

    def __init__(self, J=1, T=1, geometry=(1,), configuration=None, save_path='.'):
        self.J = J
        self.T = T
        self.geometry = geometry
        self.configuration = configuration
        self.save_path = save_path

    def random_configuration(self):
        """ Distribute particles according to random configuration """
        configuration = np.random.choice([-1, 1], size=self.geometry)
        self.configuration = configuration

    def uniform_configuration(self):
        """ Distribute particles according to uniform configuration """
        configuration = np.ones(self.geometry)
        self.configuration = configuration

    def generate_ensemble(self, ensemble_size, eq=True):
        """ Equilibrate configuration to T and run MCMC until ensemble_size is reached """
        if self.configuration is None:
            self.random_configuration()

        if eq:
            self.configuration = run_mcmc(self.J, self.T, self.configuration)
        self.ensemble, self.energies = run_mcmc(self.J, self.T, self.configuration, ensemble_size)

    def save_model(self, name='model.pkl'):
        file_out = os.path.join(self.save_path, name)
        if os.path.exists(file_out):
            raise ValueError('model with this name already exists')
        with open(file_out, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, name='model.pkl'):
        if not os.path.exists(name):
            raise ValueError('model does not exists')
        with open(name, 'rb') as f:
            obj = pickle.load(f)
        for key in obj.__dict__:
            setattr(self, key, obj.__dict__[key])
