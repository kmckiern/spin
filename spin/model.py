import os

import numpy as np
import copy
from pprint import pprint
import pickle

from spin.system import System
from spin.ensemble import Ensemble
from spin.network import Hopfield, RestrictedBoltzmann, VAE
from spin.plot import plot_ensemble, plot_rbm


class Model(object):

    """ Create, equilibrate, measure, and build network of model """

    def __init__(self):
        self.system = None
        self.ensemble = None
        self.network = None

    def generate_system(self, T=1, spin=1, geometry=(1,), configuration=None,
                        save_path=None):
        self.system = System(T, spin, geometry, configuration)
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def generate_ensemble(self, n_samples=1):
        self.ensemble = Ensemble(self.system, n_samples)

    def generate_RBM(self, optimize=None):
        self.network = RestrictedBoltzmann(self, optimize)

    def describe(self, s_obj):
        if s_obj == None:
            raise ValueError('object has not yet been created')
        system_properties = s_obj.__dict__
        pprint(system_properties)

    def describe_system(self):
        self.describe(self.system)

    def describe_ensemble(self):
        self.describe(self.ensemble)
        plot_ensemble(self)

    def describe_network(self):
        self.describe(self.network)
        plot_rbm(self)

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
