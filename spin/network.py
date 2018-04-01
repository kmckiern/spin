import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
import torch.nn as nn


class Network(object):

    """ Base class for all network models """

    def __init__(self, model, split_ratio=.8, flatten=True):

        data = model.ensemble.configuration
        self.n_samples = data.shape[0]

        if data.ndim == 2:
            self.n_neurons = data.shape[-1]
        elif data.ndim == 3:
            nr, nc = data.shape[1:]
            self.n_neurons = nr*nc
            if flatten:
                data = data.reshape(self.n_samples, self.n_neurons)

        self.n_hidden = int(self.n_neurons * .5)

        self.data = data
        self.split_ratio = split_ratio
        self.train_data, self.test_data = train_test_split(self.data,
                train_size=self.split_ratio)


class RestrictedBoltzmann(Network):

    """
    Restricted Boltzmann Machine (RBM) network model
    min(KL(P_h||P_v))
    """

    def __init__(self, model, optimize=False):

        super(RestrictedBoltzmann, self).__init__(model)

        if optimize:
            batch_size = [2**i for i in range(2, int(self.n_hidden**.5)+1)]
            learning_rate = [0.01, .001, .0001, .00001]
            n_iter = [10, 100, 1000]
            hypers = {'batch_size': batch_size,
                      'learning_rate': learning_rate,
                      'n_iter': n_iter}
        else:
            hypers = None

        self.build(optimize_h=hypers)

    def optimize_hyperparams(self, hyper_dict):

        """ Optimize hyperparams, score using pseudo-likelihood """

        import itertools

        hyper_ps = sorted(hyper_dict)
        combs = list(itertools.product(*(hyper_dict[name] for name in hyper_ps)))
        scores = {}
        for cndx, c in enumerate(combs):
            sub_dict = dict(zip(hyper_ps, c))
            rbm = BernoulliRBM(n_components=self.n_hidden, verbose=True,
                               **sub_dict)
            rbm.fit(self.train_data)
            score = np.sum(rbm.score_samples(self.test_data))
            scores[score] = rbm

        best_score = max(scores.keys())
        self.rbm = scores[best_score]

    def build(self, optimize_h=None):
    
        """ Train weights via contrastive divergence """

        if optimize_h == None:
            rbm = BernoulliRBM(n_components=self.n_hidden, verbose=True)
            rbm.fit(self.train_data)
            self.rbm = rbm
        else:
            self.optimize_hyperparams(optimize_h)

