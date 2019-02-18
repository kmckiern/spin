import numpy as np
from sklearn.neural_network import BernoulliRBM

from spin.networks.network import Network
from spin.networks.vae import VAE


def test_vae_init():
    ensemble = np.load('resources/high_T_4x4_ensemble_5000.npy')
    vae = Network(ensemble, VAE)

    assert vae.train.shape == (3000, 16)
    assert vae.test.shape == (1000, 16)
    assert vae.valid.shape == (1000, 16)


def test_vae_fit():
    ensemble = np.load('resources/high_T_4x4_ensemble_5000.npy')
    vae = Network(ensemble, VAE)
    vae.fit()

    assert hasattr(vae, 'scores')
    assert hasattr(vae, 'model')
