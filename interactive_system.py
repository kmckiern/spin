#!/bin/env python

from spin import *
import IPython

x = system(spin=1, geometry=(4,6))
x.random_configuration()

IPython.embed()

