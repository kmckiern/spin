import numpy as np
from sklearn.neural_network import BernoulliRBM

from spin.networks.network import Network


def test_rbm_init():
    ensemble = np.load('resources/high_T_4x4_ensemble_5000.npy')
    rbm = Network(ensemble, BernoulliRBM)

    assert rbm.train.shape == (3000, 16)
    assert rbm.test.shape == (1000, 16)
    assert rbm.valid.shape == (1000, 16)


def test_rbm_fit():
    ensemble = np.load('resources/high_T_4x4_ensemble_5000.npy')
    rbm = Network(ensemble, BernoulliRBM)
    rbm.fit()

    assert hasattr(rbm, 'scores')
    assert hasattr(rbm, 'model')
