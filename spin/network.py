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

    def split(self, data, split_ratio):

        """
        ratio = test/train divide of data
        """

        n_samples = data.shape[0]
        divide = int(n_samples * split_ratio)
        train = data[:divide]
        test = data[divide:]
        return train, test

    def random_split(self, data, split_ratio):

        """
        Randomizes data, then splits
        """

        n_samples = data.shape[0]
        mix_ndx = np.random.permutation(n_samples)
        mixed_data = data[mix_ndx]
        return self.split(mixed_data, split_ratio)

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
        n_samples, nr, nc = data.shape
        # easier to do this using 1D array of values
        data = data.reshape(n_samples, nr*nc)
        self.data = data
        self.train, self.test = self.split(data, split_ratio)
        self.weights = self.fit(self.train)

    def fit(self, training_data):

        """
        Use each train sample to construct the weight matrix
        Update weights according to generalized Hebbian rule
        """

        n_neurons = len(training_data[0])
        weights = np.zeros((n_neurons, n_neurons))
        for sample in training_data:
            weights += np.outer(sample, sample)
        # normalize, and zero diagonal
        weights /= n_neurons
        np.fill_diagonal(weights, 0)
        return weights

    def test(self, weights, data):

        """
        Use each sample to test the weight matrix
        """
        
