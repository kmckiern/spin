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
        self.weights = np.zeros(data[0].shape)

    def weight_update(self, weights):

        """
        Update weights according to generalized Hebbian rule
        """

        rows, cols = configuration.shape
        for r in rows:
            for c in cols:
                if r <= c:
                    continue
                w_rc = (2*configuration[i,:] - 1) * (2*configuration[:,j] - 1) 
                weights[r][c] = w_rc
        return weights

    def fit(self, weights, data):

        """
        Use each sample to train the weight matrix
        """

        for samples in data:
            weights += weight_update(sample)
        weights += weights.T
        weights /= weights.size
        return weights

    def test(self, weights, data):

        """
        Use each sample to test the weight matrix
        """
        
    def build(self):
        train, test = self.split()
        weights = self.fit(self.weights, train)
        self.weights = weights
        score = self.test(weights, test)
        print (score)
