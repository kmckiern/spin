import os

import numpy as np
import copy
from pprint import pprint
import pickle

from spin.system import System
from spin.ensemble import Ensemble
from spin.network import Hopfield, RestrictedBoltzmann
from spin.plot import plot_ensemble


class Model(object):

    """ Create, equilibrate, measure, and build network of model """

    def __init__(self):
        self.system = None
        self.ensemble = None
        self.network = None

    def generate_system(self, T=1, spin=1, geometry=(1,), configuration=None):
        self.system = System(T, spin, geometry, configuration)

    def generate_ensemble(self, n_samples=1):
        self.ensemble = Ensemble(self.system, n_samples)

    def generate_hopfield(self):
        self.network = Hopfield(self.ensemble.configuration)

    def generate_RBM(self):
        self.network = RestrictedBoltzmann(self.ensemble.configuration)

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

    def save_model(self, name='model.pkl'):
        if os.path.exists(name):
            raise ValueError('model with this name already exists')
        with open(name, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, name='model.pkl'):
        if not os.path.exists(name):
            raise ValueError('model does not exists')
        with open(name, 'rb') as f:
            obj = pickle.load(f)
            obj.save_path = name.split('.pkl')[0]
        for key in obj.__dict__:
            setattr(self, key, obj.__dict__[key])
