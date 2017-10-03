import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from .operators import Operators


class Network(Operators):

    """ Base class for all network models """

    def __init__(self, data, split_ratio=.8):

        self.n_samples, nr, nc = data.shape
        self.n_neurons = nr*nc

        # easier to do this using 1D array of values
        data = data.reshape(self.n_samples, self.n_neurons)
        self.data = data

        self.split_ratio = split_ratio
        self.train_data, self.test_data = train_test_split(self.data,
                train_size=self.split_ratio)


class RestrictedBoltzmann(Network):

    """ Restricted Boltzmann Machine (RBM) network model

    min(KL(P_h||P_v))
    """

    def __init__(self, data, split_ratio=.8, 
            batch_size = None,
            learning_rate = [0.1, 0.01, .001],
            n_iter = [100, 1000, 10000]):

        self.rbm = None

        super(RestrictedBoltzmann, self).__init__(data, split_ratio)

        self.n_hidden = int(self.n_neurons * .5)
        if batch_size is None:
            batch_size = [2**i for i in range(2, int(self.n_hidden**.5)+1)]

        self.hypers = {'batch_size': batch_size,
                  'learning_rate': learning_rate,
                  'n_iter': n_iter}

        self.build(optimize_h=True)

    def optimize_hyperparams(self):

        """ Optimize hyperparams, score using pseudo-likelihood """

        import itertools

        hyper_ps = sorted(self.hypers)
        combs = list(itertools.product(*(self.hypers[name] for name in hyper_ps)))
        scores = {}
        for cndx, c in enumerate(combs):
            sub_dict = dict(zip(hyper_ps, c))
            rbm = BernoulliRBM(n_components=self.n_hidden, **sub_dict)
            rbm.fit(self.train_data)
            score = np.sum(rbm.score_samples(self.test_data))
            scores[score] = sub_dict
        best_score = max(scores.keys())
        self.hypers = scores[best_score]

    def build(self, optimize_h=False):
    
        """ Train weights via contrastive divergence """

        if optimize_h:
            self.optimize_hyperparams()
        rbm = BernoulliRBM(n_components=self.n_hidden, verbose=True,
                **self.hypers)
        rbm.fit(self.train_data)
        self.rbm = rbm
