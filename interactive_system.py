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

# generate ensemble from system
x.generate_ensemble(n_samples=8)

# view ensemble configurations as np array
ec = x.ensemble.configuration

# print descriptions of system
# x.describe_system()

# create hopfield network
# x.generate_hopfield()

# create RMB
# x.generate_RBM()

# start interactive session to work with system/ensemble
IPython.embed()
