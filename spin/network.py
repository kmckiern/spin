import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from .operators import Operators


def conditional_prob(a, b, epsilon):

    """ P(a_i|b) """

    return np.dot(a, b) + epsilon

def sigmoid(x):

    """ Common activation function, strangely not in numpy afaik """

    return 1 / (1 + np.exp(-x))

def binary_sig_prob(probs):

    """ Apply activation to probabilities, and threshold """

    activated = np.round(sigmoid(probs))
    activated[activated == 0] = -1
    return activated


class Network(Operators):

    """ Base class for all network models """

    def __init__(self, data, split_ratio=.8):

        self.n_samples, nr, nc = data.shape
        self.n_neurons = nr*nc

        # easier to do this using 1D array of values
        data = data.reshape(self.n_samples, self.n_neurons)
        self.data = data

        self.split_ratio = split_ratio
        self.train_data, self.test_data = train_test_split(self.data,
                train_size=self.split_ratio)

class Hopfield(Network):

    """
    Hopfield network model
    doi:10.1073/pnas.79.8.2554 
    http://web.cs.ucla.edu/~rosen/161/notes/hopfield.html

    Parameters
    ----------
    data : binary arrays of spin configurations
    split_ratio : test/train ratio
    """

    def __init__(self, data, split_ratio=.8):

        super(Hopfield, self).__init__(data, split_ratio)
        self.weights = np.zeros((self.n_neurons, self.n_neurons))

        self.train()
        self.test_errors = self.test()

    def train(self):

        """ Train weights according to generalized Hebbian rule """

        for sample in self.train_data:
            self.weights += np.outer(sample, sample)
        # normalize, and zero diagonal
        self.weights /= len(self.train_data)
        np.fill_diagonal(self.weights, 0)

    def test(self):

        """ Use each sample to synchronously test the weight matrix """
        
        recall = np.sign(np.dot(self.test_data, self.weights))
        return np.sum(recall != self.test_data, axis=1)

class RestrictedBoltzmann(Network):

    """ Restricted Boltzmann Machine (RBM) network model

    min(KL(P_h||P_v))
    """

    def __init__(self, data, split_ratio=.8, learning_rate=[0.1, 0.01, .001],
	n_hidden=None, n_iter=None):

        super(RestrictedBoltzmann, self).__init__(data, split_ratio)

        self.learning_rate = learning_rate
        if n_hidden is None:
            n_hidden = int(self.n_neurons * .5)
        self.n_hidden = n_hidden
        self.n_iter = n_iter 

        self.rbm = self.build()

    def build(self, error_threshold=1.1):
    
        """ Train weights via contrastive divergence """

        rbm = BernoulliRBM(n_components=self.n_hidden, verbose=True)
        rbm.fit(self.train_data)
        return rbm
