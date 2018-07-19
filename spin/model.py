import os

from pprint import pprint
import pickle

from spin.ensemble import Ensemble
from spin.network import RestrictedBoltzmann, VAE
from spin.plot import plot_ensemble, plot_rbm, plot_train, plot_reconstruction


class Model:
    """ Create, equilibrate, measure, and build network of model """
    def __init__(self,
                 T=1,
                 spin=1,
                 geometry=(1,),
                 configuration=None,
                 save_path='.'):

        self.T = T
        self.spin = spin
        self.geometry = geometry
        self.configuration = configuration

        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def random_configuration(self):
        """ Distribute particles according to random configuration """
        configuration = np.random.choice([-1, 1], size=self.geometry)
        configuration *= self.spin
        self.configuration = configuration

    def uniform_configuration(self, val=1):
        """ Distribute particles according to uniform configuration """
        configuration = np.ones(self.geometry)
        configuration *= self.spin
        self.configuration = configuration

    def generate_ensemble(self, n_samples=1, configurations=None):
        self.ensemble = Ensemble(self.system, n_samples, configurations)

    def generate_RBM(self, n_hidden=None, lr=.01, batch_size=64, n_iter=5, optimize=False):
        hypers = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'n_iter': n_iter
        }
        hypers = correct_hyper_dict(hypers, optimize)

        self.RBM = RestrictedBoltzmann(self, n_hidden, hypers, optimize)

    def generate_VAE(self, n_hidden=None, lr=.01, batch_size=64, n_iter=5,
                     optimize=False):
        hypers = {
            'learning_rate': lr,
            'batch_size': batch_size,
            'n_iter': n_iter
        }
        hypers = correct_hyper_dict(hypers, optimize)

        self.VAE = VAE(self, n_hidden, hypers, optimize)

    def describe(self, component='system', plot_component=False):
        model_component = self.__dict__[component]
        component_attributes = model_component.__dict__
        pprint(component_attributes)

        if plot_component:
            if component == 'ensemble':
                plot_ensemble(self)
            elif component == 'RBM':
                plot_rbm(self)
            elif component == 'VAE':
                plot_reconstruction(self)
                plot_train(self, component)

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

