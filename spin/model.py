import numpy as np
from .system import System
from .operators import Operators
from .ensemble import Ensemble
import copy
from pprint import pprint

class Model(object):

    """
    Create, equilibrate, measure, and build network of model
    """

    def __init__(self):
        self._system = None
        self._ensemble = None

    def generate_system(self, T=1, spin=1, geometry=(1,), configuration=None):
        self._system = System(T, spin, geometry, configuration)

    def measure_system(self, J=-1.0):
        self._system._observables = Operators(self._system._configuration, J)

    def generate_ensemble(self, n_samples=1):
        self._ensemble = Ensemble(self, n_samples)

    def measure_ensemble(self, J=-1.0):
        self._ensemble._observables = Operators(self._ensemble._configurations, J)

    def describe(self, d):
        d = copy.copy(d)
        d['_observables'] = d['_observables'].__dict__
        pprint(d)

    def describe_system(self):
        system_properties = self._system.__dict__
        self.describe(system_properties)

    def describe_ensemble(self):
        ensemble_properties = self._ensemble.__dict__
        self.describe(ensemble_properties)
