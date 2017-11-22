#!/bin/env python

from spin import Model
import os
import numpy as np
import IPython

# lattice and system parameters
geo = (6,6)
temp = 3

# create model object
x = Model()

# create a system, with or without an IC
x.generate_system(geometry=geo, T=temp)

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
