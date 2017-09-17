import numpy as np
from .operators import Operators

import copy

class Network(Operators):

    """
    Base class for all network models
    """

    def __init__(self):
        pass

    def train(self, data):
        return self

    def test(self, data):
        return self

    def split(self):

        """
        ratio = test/train divide of data
        """

        divide = int(self.n_samples * self.split_ratio)
        train = self.data[:divide]
        test = self.data[divide:]
        return train, test

    def random_split(self, data, split_ratio):

        """
        Randomizes data, then splits
        """

        mix_ndx = np.random.permutation(self.n_samples)
        self.data = self.data[mix_ndx]
        self.split()

class Hopfield(Network):

    """
    Hopfield network model for input data
    doi:10.1073/pnas.79.8.2554 
    http://web.cs.ucla.edu/~rosen/161/notes/hopfield.html

    Parameters
    ----------
    data : binary arrays of spin configurations
    split : test/train ratio
    """

    def __init__(self, data, split_ratio=.8):

        self.n_samples, nr, nc = data.shape
        self.n_neurons = nr*nc
        # easier to do this using 1D array of values
        data = data.reshape(self.n_samples, self.n_neurons)

        self.data = data
        self.split_ratio = split_ratio
        self.train_data, self.test_data = self.split()
        self.weights = self.train()
        self.test_errors = self.test()

    def train(self):

        """
        Train weights according to generalized Hebbian rule
        """

        weights = np.zeros((self.n_neurons, self.n_neurons))
        for sample in self.train_data:
            weights += np.outer(sample, sample)
        # normalize, and zero diagonal
        weights /= len(self.train_data)
        np.fill_diagonal(weights, 0)
        return weights

    def test(self):

        """
        Use each sample to test the weight matrix
        """
        
        recall = np.sign(np.dot(self.test_data, self.weights))
        return np.sum(recall != x.network.test_data, axis=1)
