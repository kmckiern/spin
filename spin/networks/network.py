from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from typing import List


class Network(BaseEstimator):
    """ Abstract base class """

    def __init__(self,
                 data,
                 n_hidden = None,
                 train_percent: float = .6,
                 batch_size: List[int] = [64],
                 learning_rate: List[float] = .001,
                 n_epochs: List[int] = [100],
                 verbose: bool = False):

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

        # extra
        self.verbose = verbose

        if n_hidden is None:
            self.n_hidden = int(self.n_visible * .5)
        else:
            self.n_hidden = n_hidden

    def _fit(self, sub_dict):
        pass

    def _optimize_hyperparameters(self, hyper_ps, combs):
        pass

    def fit(self):
        pass

    def score_samples(selfself, data):
        pass
