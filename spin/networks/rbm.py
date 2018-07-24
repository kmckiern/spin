import itertools

import numpy as np
from sklearn.neural_network import BernoulliRBM

from spin.networks.network import Model


class RestrictedBoltzmann(Model):
    """ Restricted Boltzmann Machine (RBM) network model, min(KL(P_h||P_v)) """

    def __init__(self,
                 data,
                 n_hidden=None,
                 train_percent=.6,
                 batch_size=[64],
                 learning_rate=[.001],
                 n_epochs=[100],
                 verbose=True):

        super(RestrictedBoltzmann, self).__init__(data,
                                                  train_percent,
                                                  batch_size,
                                                  learning_rate,
                                                  n_epochs,
                                                  verbose)

        if n_hidden is None:
            self.n_hidden = int(self.n_visible * .5)
        else:
            self.n_hidden = n_hidden

    def _optimize_hyperparameters(self, hyper_ps, combs):
        hyper_scores = {}
        for cndx, c in enumerate(combs):
            sub_dict = dict(zip(hyper_ps, c))

            rbm = BernoulliRBM(n_components=self.n_hidden, verbose=self.verbose, **sub_dict)
            rbm.fit(self.train)

            scores = {}
            scores['train'] = np.sum(rbm.score_samples(self.train))
            scores['valid'] = np.sum(rbm.score_samples(self.valid))
            scores['test'] = np.sum(rbm.score_samples(self.test))
            hyper_scores[scores['valid']] = (scores, rbm)

        best_score = min(hyper_scores.keys())
        self.scores, self.rbm = hyper_scores[best_score]

    def fit(self):
        hyper_ps = sorted(self.hyperparameters)
        combs = list(itertools.product(*(self.hyperparameters[name] for name in hyper_ps)))

        if self.optimize:
            self._optimize_hyperparameters(hyper_ps, combs)
        else:
            sub_dict = dict(zip(hyper_ps, combs[0]))

            rbm = BernoulliRBM(n_components=self.n_hidden, verbose=True, **sub_dict)
            rbm.fit(self.train)
            self.rbm = rbm

            self.scores = {}
            self.scores['train'] = np.sum(rbm.score_samples(self.train))
            self.scores['valid'] = np.sum(rbm.score_samples(self.valid))
            self.scores['test'] = np.sum(rbm.score_samples(self.test))
