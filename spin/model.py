import os
import pickle

import numpy as np
from sklearn.neural_network import BernoulliRBM

from spin.ensemble import run_mcmc
from spin.operators import measure_magnetization
from spin.networks.network import Network
from spin.networks.vae import VAE


class Model:
    """ Create, equilibrate, measure, and build network of model """

    def __init__(self, J=1, T=1, geometry=(1,), configuration=None, save_path='.', model_file=None):
        self.J = J
        self.T = T

        self.geometry = geometry
        self.configuration = configuration

        self.save_path = save_path

        if model_file is not None:
            self.load_model(model_file)

        self.RBM = None
        self.VAE = None

    def random_configuration(self):
        """ Distribute particles according to random configuration """
        configuration = np.random.choice([-1, 1], size=self.geometry)
        self.configuration = configuration

    def uniform_configuration(self):
        """ Distribute particles according to uniform configuration """
        configuration = np.ones(self.geometry)
        self.configuration = configuration

    def generate_ensemble(self, ensemble_size, eq=True, autocorrelation_threshold=.1):
        """ Equilibrate configuration to T and run MCMC until ensemble_size is reached """
        if self.configuration is None:
            self.random_configuration()

        if eq:
            self.configuration = run_mcmc(self.J, self.T, self.configuration)

        self.ensemble, self.energies = run_mcmc(self.J, self.T, self.configuration, desired_samples=ensemble_size,
                                                autocorrelation_threshold=autocorrelation_threshold)

        self.magnetization = []
        for configuration in self.ensemble:
            self.magnetization.append(measure_magnetization(configuration))
        self.magnetization = np.array(self.magnetization)

    def generate_rbm(self):
        if not hasattr(self, 'ensemble'):
            raise ValueError('must first load or generate ensemble')

        rbm = Network(self.ensemble, BernoulliRBM)
        rbm.fit()

        self.RBM = rbm.model

    def generate_vae(self):
        if not hasattr(self, 'ensemble'):
            raise ValueError('must first load or generate ensemble')

        vae = Network(self.ensemble, VAE)
        vae.fit()

        self.VAE = vae.model

    def save_model(self, name='model.pkl'):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

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
