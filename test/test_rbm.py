import numpy as np

from spin.networks.rbm import RestrictedBoltzmann


def test_rbm_init():
    ensemble = np.load('resources/high_T_4x4_ensemble_5000.npy')
    rbm_model = RestrictedBoltzmann(ensemble)

    assert rbm_model.train.shape == (3000, 16)
    assert rbm_model.test.shape == (1000, 16)
    assert rbm_model.valid.shape == (1000, 16)


def test_rbm_fit():
    ensemble = np.load('resources/high_T_4x4_ensemble_5000.npy')
    rbm_model = RestrictedBoltzmann(ensemble)
    rbm_model.fit()

    assert hasattr(rbm_model, 'scores')
    assert hasattr(rbm_model, 'rbm')
