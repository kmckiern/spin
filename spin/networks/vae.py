import itertools

import torch.nn as nn

from spin.networks.network import Model


class VAE(Model, nn.Module):
    """ Variational autoencoder network model, min(KL(P_h||P_v)) """

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

        self.encoder = nn.Sequential(
            nn.Linear(self.n_visible, self.n_hidden))
        self.decoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_visible),
            nn.Sigmoid())

    def _fit(self, learning_rate, batch_size, n_iter):
        train_data = torch.from_numpy(self.train).float()
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        compute_loss = nn.MSELoss()
        for epoch in range(n_iter):
            for step, x in enumerate(train_loader):
                reference = Variable(x.view(-1, self.n_visible))

                b_x = Variable(x.view(-1, self.n_visible))
                encoded = self.encoder(b_x)
                decoded = self.decoder(encoded)
                error = compute_loss(decoded, reference)

                optimizer.zero_grad()
                error.backward()
                optimizer.step()

        return error.data[0]

    def _optimize_hyperparameters(self, hyper_ps, combs):
        hyper_scores = {}
        for cndx, c in enumerate(combs):
            sub_dict = dict(zip(hyper_ps, c))
            hyper_scores[cndx] = self._fit(**sub_dict)

        best_score = min(hyper_scores.values())
        self.vae = hyper_scores[best_score]

    def fit(self):
        hyper_ps = sorted(self.hyperparameters)
        combs = list(itertools.product(*(self.hyperparameters[name] for name in hyper_ps)))

        if self.optimize:
            self._optimize_hyperparameters(hyper_ps, combs)
        else:
            sub_dict = dict(zip(hyper_ps, combs[0]))
            self.vae = self._fit(**sub_dict)
