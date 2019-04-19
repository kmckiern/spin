import itertools
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


class Network(BaseEstimator):
    def __init__(self,
                 data,
                 model_class,
                 n_hidden=None,
                 batch_size=None,
                 learning_rate=None,
                 n_epochs=None,
                 train_percent: float = .6,
                 verbose: bool = False):

        # avoid mutable defaults
        if batch_size is None:
            batch_size = [64]
        if learning_rate is None:
            learning_rate = [.001]
        if n_epochs is None:
            n_epochs = [100]

        # reshape data
        if data.ndim == 3:
            n_samples, n_row, n_col = data.shape
            self.n_visible = n_row * n_col
            data = data.reshape(n_samples, self.n_visible)

        # split data
        self.train, test_valid = train_test_split(data, train_size=train_percent)
        self.test, self.valid = train_test_split(test_valid, train_size=.5)

        # learning parameters
        self.hyperparameters = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'n_iter': n_epochs
        }
        if (len(learning_rate) > 1) or (len(batch_size) > 1) or (len(n_epochs) > 1):
            self.optimize = True
        else:
            self.optimize = False

        if n_hidden is None:
            self.n_hidden = int(self.n_visible * .5)
        else:
            self.n_hidden = n_hidden

        self.model_class = model_class

        # extra
        self.verbose = verbose

    def _fit(self, sub_dict):
        model = self.model_class(n_components=self.n_hidden, verbose=self.verbose, **sub_dict)
        model.fit(self.train)

        scores = {}
        scores['train'] = np.sum(model.score_samples(self.train))
        scores['valid'] = np.sum(model.score_samples(self.valid))
        scores['test'] = np.sum(model.score_samples(self.test))

        return scores, model

    def _optimize_hyperparameters(self, hyper_ps, combs):
        hyper_scores = {}
        for cndx, c in enumerate(combs):
            sub_dict = dict(zip(hyper_ps, c))
            scores, model = self._fit(sub_dict)
            hyper_scores[scores['valid']] = (scores, model)

        best_score = min(hyper_scores.keys())
        self.scores, self.model = hyper_scores[best_score]

    def fit(self):
        hyper_ps = sorted(self.hyperparameters)
        combs = list(itertools.product(*(self.hyperparameters[name] for name in hyper_ps)))

        if self.optimize:
            self._optimize_hyperparameters(hyper_ps, combs)
        else:
            sub_dict = dict(zip(hyper_ps, combs[0]))
            self.scores, self.model = self._fit(sub_dict)
