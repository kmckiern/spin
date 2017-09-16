#!/bin/env python

from spin import Model
import os
import numpy as np
import IPython

# lattice and system parameters
geo = (6,6)
temp = 3
example_IC = os.path.join('.', 'samples/config.npy')

# create model object
x = Model()

# create a system, with or without an IC
if os.path.exists(example_IC):
    x.generate_system(geometry=geo, T=temp, configuration=np.load(example_IC))
else:
    x.generate_system(geometry=geo, T=temp)
    x.random_configuration()

# measure observables of created system
x.measure_system()

# generate ensemble from system
x.generate_ensemble(n_samples=5)

# view ensemble configurations as np array
ec = x._ensemble._configurations

# measure observables on ensemble
x.measure_ensemble()

# print descriptions of state and ensemble
print('~ system description ~')
x.describe_system()
print ('\n')
print('~ ensemble description ~')
x.describe_ensemble()

# create hopfield network
x.generate_hopfield()

# print description of network
print ('~ network description ~')
x.describe_network()

# start interactive session to work with system/ensemble
IPython.embed()
