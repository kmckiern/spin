import itertools

import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

from spin.networks.network import Model


class AutoEncoder(nn.Module):
    """ Variational autoencoder network model, min MS reconstruction E """

    def __init__(self,
                 n_visible,
                 n_hidden=None,
                 batch_size=64,
                 learning_rate=.001,
                 n_iter=100):

        super(AutoEncoder, self).__init__()

        self.n_visible = n_visible
        if n_hidden is None:
            self.n_hidden = int(self.n_visible * .5)
        else:
            self.n_hidden = n_hidden

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_iter = n_iter

        self.encoder = nn.Sequential(
            nn.Linear(self.n_visible, self.n_hidden))
        self.decoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_visible),
            nn.Sigmoid())

    def fit(self, training_data):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        compute_loss = nn.MSELoss()

        train_data = torch.from_numpy(training_data).float()
        train_loader = Data.DataLoader(dataset=train_data,
                                       batch_size=self.batch_size)

        for epoch in range(self.n_iter):
            for step, x in enumerate(train_loader):
                b_x = Variable(x.view(-1, self.n_visible))

                encoded = self.encoder(b_x)
                decoded = self.decoder(encoded)

                error = compute_loss(decoded, b_x)

                optimizer.zero_grad()
                error.backward()
                optimizer.step()

        self.score = error.data[0]

        return self.score

    def score_samples(self, data):
        x = torch.from_numpy(data).float()

        b_x = Variable(x.view(-1, self.n_visible))
        encoded = self.encoder(b_x)
        decoded = self.decoder(encoded)

        compute_loss = nn.MSELoss()
        error = compute_loss(decoded, b_x)

        return error.data[0]


class VAE(Model):

    def __init__(self,
                 data,
                 n_hidden=None,
                 train_percent=.6,
                 batch_size=[64],
                 learning_rate=[.001],
                 n_epochs=[100],
                 verbose=True):

        super(VAE, self).__init__(data,
                                  train_percent,
                                  batch_size,
                                  learning_rate,
                                  n_epochs,
                                  verbose)

        if n_hidden is None:
            self.n_hidden = int(self.n_visible * .5)
        else:
            self.n_hidden = n_hidden

    def _fit(self, sub_dict):
        vae = AutoEncoder(n_visible=self.n_visible, n_hidden=self.n_hidden, **sub_dict)
        vae.fit(self.train)

        scores = {}
        scores['train'] = vae.score_samples(self.train)
        scores['valid'] = vae.score_samples(self.valid)
        scores['test'] = vae.score_samples(self.test)

        return scores, vae

    def _optimize_hyperparameters(self, hyper_ps, combs):
        hyper_scores = {}
        for cndx, c in enumerate(combs):
            sub_dict = dict(zip(hyper_ps, c))
            scores, vae = self._fit(sub_dict)
            hyper_scores[scores['valid']] = (scores, vae)

        best_score = min(hyper_scores.keys())
        self.scores, self.vae = hyper_scores[best_score]

    def fit(self):
        hyper_ps = sorted(self.hyperparameters)
        combs = list(itertools.product(*(self.hyperparameters[name] for name in hyper_ps)))

        if self.optimize:
            self._optimize_hyperparameters(hyper_ps, combs)
        else:
            sub_dict = dict(zip(hyper_ps, combs[0]))
            self.scores, self.vae = self._fit(sub_dict)
