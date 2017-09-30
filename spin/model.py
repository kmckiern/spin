from .system import System
from .ensemble import Ensemble
from .network import Hopfield, RestrictedBoltzmann

import numpy as np
import copy
from pprint import pprint


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

    def describe_network(self):
        self.describe(self.network)

