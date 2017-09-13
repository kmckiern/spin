#!/bin/env python

from spin import Model
import IPython

x = Model()
x.generate_system(geometry=(6,6))
x.measure_system()
x.generate_ensemble()

IPython.embed()
