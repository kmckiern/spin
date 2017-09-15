import numpy as np
from .operators import Operators

class Network(Operators):

    """
    Base class for all network models
    """

    def __init__(self):
        pass

    def fit(self, data):
        return self

    def split(self, data, ratio):
        n_samples = data.shape[0]
        divide = int(n_samples * ratio)
        train = mixed_data[:divide]
        test = mixed_data[divide:]
        return train, test

    def random_split(self, data, ratio):
        n_samples = data.shape[0]
        mix_ndx = np.random.permutation(n_samples)
        mixed_data = data[mix_ndx]
        return self.split(mixed_data, ratio)

class Hopfield(Network):

    """
    Hopfield network model for input data
    doi:10.1073/pnas.79.8.2554 
    http://web.cs.ucla.edu/~rosen/161/notes/hopfield.html

    Parameters
    ----------
    data : binary arrays of spin configurations
    split : test/train ratio
    weight : model memory weight matrix
    """

    def __init__(self, data, split=.8):
        self.data = data
        self.split = split
        self.weight = np.zeros(data[0].shape)

    def weight_update(self, configuration):
        rows, cols = configuration.shape
        for r in rows:
            for c in cols:
                sample_weights[r][c] = (2*configuration[i,:] - 1) *
                    (2*configuration[:,j] - 1) 
        return weights

    def train_weights(self, data):
        for samples in data:
            self.weights += self.weight_update(sample)
        np.fill_diagonal(self.weights, 0)

