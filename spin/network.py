import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data


class Network(object):

    """ Base class for all network models """

    def __init__(self, model, n_hidden=None, split_ratio=.8, flatten=True):

        data = model.ensemble.configuration
        n_dim = model.system.n_dim
        self.n_dim = n_dim
        self.n_samples = data.shape[0]

        if n_dim == 1:
            self.n_visible = data.shape[-1]
        elif n_dim == 2:
            nr, nc = data.shape[1:]
            self.n_visible = nr*nc
            if flatten:
                data = data.reshape(self.n_samples, self.n_visible)

        if n_hidden is None:
            self.n_hidden = int(self.n_visible * .5)
        else:
            self.n_hidden = n_hidden

        self.split_ratio = split_ratio
        self.train_data, self.test_data = train_test_split(data,
                train_size=self.split_ratio)


class RestrictedBoltzmann(Network):

    """
    Restricted Boltzmann Machine (RBM) network model
    min(KL(P_h||P_v))
    """

    def __init__(self, model, hypers, optimize):

        super(RestrictedBoltzmann, self).__init__(model)

        self.hypers = hypers
        self.build(optimize)

    def optimize_hyperparams(self):

        import itertools

        scores = {}

        hyper_ps = sorted(self.hypers)
        combs = \
            list(itertools.product(*(self.hypers[name] for name in hyper_ps)))
        for cndx, c in enumerate(combs):
            sub_dict = dict(zip(hyper_ps, c))

            rbm = BernoulliRBM(n_components=self.n_hidden, verbose=True,
                               **sub_dict)
            rbm.fit(self.train_data)
            score = np.sum(rbm.score_samples(self.test_data))
            scores[score] = rbm


        self.trained_models = scores

        best_score = min(scores.keys())
        self.rbm = scores[best_score]

    def build(self, optimize):

        if optimize:
            self.optimize_hyperparams()
        else:
            rbm = BernoulliRBM(n_components=self.n_hidden, verbose=True,
                               **self.hypers)
            rbm.fit(self.train_data)
            self.rbm = rbm


class AutoEncoder(nn.Module):

    def __init__(self, n_visible, n_hidden, learning_rate, batch_size, n_iter):

        super(AutoEncoder, self).__init__()

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_iter = n_iter

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.encoder = nn.Sequential(
            nn.Linear(n_visible, n_hidden))
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, n_visible),
            nn.Sigmoid())

    def fit(self, training_data, n_track=4):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        compute_loss = nn.MSELoss()

        train_data = torch.from_numpy(training_data).float()
        train_loader = Data.DataLoader(dataset=train_data,
                                       batch_size=self.batch_size)

        train_log = {}
        for epoch in range(self.n_iter):
            for step, x in enumerate(train_loader):
                reference = Variable(x.view(-1, self.n_visible))

                b_x = Variable(x.view(-1, self.n_visible))
                encoded = self.encoder(b_x)
                decoded = self.decoder(encoded)

                error = compute_loss(decoded, reference)

                optimizer.zero_grad()
                error.backward()
                optimizer.step()

                if step == 0:
                    l_error = error.data.numpy()[0]
                    l_encode = encoded.data.numpy()[:n_track]
                    l_decode = decoded.data.numpy()[:n_track]
                    if epoch == 0:
                        for ele in ['error', 'encoded', 'decoded']:
                            train_log[ele] = []
                        l_ref = reference.data.numpy()[:n_track]
                        train_log['reference'] = l_ref
                    train_log['error'].append(l_error)
                    train_log['encoded'].append(l_encode)
                    train_log['decoded'].append(l_decode)

        self.train_log = train_log
        self.score = error.data[0]


class VAE(Network):

    def __init__(self, model, n_hidden, hypers, optimize):

        super(VAE, self).__init__(model, n_hidden, flatten=False)

        self.hypers = hypers
        self.build(optimize)

    def optimize_hyperparams(self):

        import itertools

        scores = {}

        hyper_ps = sorted(self.hypers)
        combs = \
            list(itertools.product(*(self.hypers[name] for name in hyper_ps)))
        for cndx, c in enumerate(combs):
            sub_dict = dict(zip(hyper_ps, c))

            vae = AutoEncoder(self.n_visible, self.n_hidden, **sub_dict)
            vae.fit(self.train_data)

            score = vae.score
            scores[score] = vae

        self.trained_models = scores

        best_score = min(scores.keys())
        self.vae = scores[best_score]

    def build(self, optimize):

        if optimize:
            self.optimize_hyperparams()
        else:
            vae = AutoEncoder(self.n_visible, self.n_hidden, **self.hypers)
            vae.fit(self.train_data)
            self.vae = vae
