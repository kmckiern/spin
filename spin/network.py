import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data


class Network(object):

    """ Base class for all network models """

    def __init__(self, model, split_ratio=.8, flatten=True):

        data = model.ensemble.configuration
        self.n_samples = data.shape[0]

        if data.ndim == 2:
            self.n_visible = data.shape[-1]
        elif data.ndim == 3:
            nr, nc = data.shape[1:]
            self.n_visible = nr*nc
            if flatten:
                data = data.reshape(self.n_samples, self.n_visible)

        self.n_hidden = int(self.n_visible * .5)

        self.split_ratio = split_ratio
        self.train_data, self.test_data = train_test_split(data,
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


class AutoEncoder(nn.Module):

    def __init__(self, n_visible=4, n_hidden=2, learning_rate=.01,
                 batch_size=10, n_epochs=50):

        super(AutoEncoder, self).__init__()

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.encoder = nn.Sequential(
            nn.Linear(n_visible, n_hidden))
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, n_visible),
            nn.Sigmoid())

    def fit(self, train_data):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        compute_loss = nn.MSELoss()

        train_data = torch.from_numpy(train_data).float()
        train_loader = Data.DataLoader(dataset=train_data,
                                       batch_size=self.batch_size,
                                       shuffle=True)

        for epoch in range(self.n_epochs):
            for step, x in enumerate(train_loader):
                b_x = Variable(x.view(-1, self.n_visible))
                reference = Variable(x.view(-1, self.n_visible))
                encoded = self.encoder(b_x)
                decoded = self.decoder(encoded)
                error = compute_loss(decoded, reference)
                optimizer.zero_grad()
                error.backward()
                optimizer.step()

        self.score = error.data[0]


class VAE(Network):

    def __init__(self, model, optimize=False):

        super(VAE, self).__init__(model)

        if optimize:
            batch_size = [2**i for i in range(2, int(self.n_hidden**.5)+1)]
            learning_rate = [0.01, .001]
            n_epochs = [100, 1000]
            hypers = {'batch_size': batch_size,
                      'learning_rate': learning_rate,
                      'n_epochs': n_epochs}
        else:
            hypers = None

        self.build(optimize_h=hypers)

    def optimize_hyperparams(self, hyper_dict):

        import itertools

        hyper_ps = sorted(hyper_dict)
        combs = list(itertools.product(*(hyper_dict[name] for name in hyper_ps)))
        scores = {}
        for cndx, c in enumerate(combs):
            sub_dict = dict(zip(hyper_ps, c))
            vae = AutoEncoder(self.n_visible, self.n_hidden, **sub_dict)
            vae.fit(self.train_data)
            score = vae.score
            scores[score] = vae
            print(c, score)

        best_score = min(scores.keys())
        self.vae = scores[best_score]

    def build(self, optimize_h=None):

        if optimize_h == None:
            vae = AutoEncoder(self.n_visible, self.n_hidden)
            vae.fit(self.train_data)
            self.vae = vae
        else:
            self.optimize_hyperparams(optimize_h)
