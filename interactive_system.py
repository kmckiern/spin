#!/bin/env python

from spin import Model
import IPython

x = Model()

x.generate_system(geometry=(6,6), T=3, configuration=np.load('samples/config.npy'))
x.measure_system()

x.generate_ensemble(n_samples=5)
ec = x._ensemble._configurations
x.measure_ensemble()

IPython.embed()
