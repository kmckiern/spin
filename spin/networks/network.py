from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from typing import List


class Model(BaseEstimator):
    """ Abstract base class """

    def __init__(self,
                 data,
                 train_percent: float,
                 batch_size: List[int],
                 learning_rate: List[float],
                 n_epochs: List[int],
                 verbose: bool):

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

    def _optimize_hyperparameters(self):
        pass

    def fit(self):
        pass
