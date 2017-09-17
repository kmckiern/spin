from .system import System
from .ensemble import Ensemble
from .network import Hopfield

import numpy as np
import copy
from pprint import pprint


class Model(object):

    """
    Create, equilibrate, measure, and build network of model
    """

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

    def describe(self, d):
        pprint(d)

    def describe_system(self):
        system_properties = self.system.__dict__
        self.describe(system_properties)

    def describe_ensemble(self):
        ensemble_properties = self.ensemble.__dict__
        self.describe(ensemble_properties)

    def describe_network(self):
        network_properties = self.network.__dict__
        self.describe(network_properties)

