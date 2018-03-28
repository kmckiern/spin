#!/bin/env python

from spin import Model
import IPython

# load model object
x = Model()
x.load_model('../samples/unit/T3.16_(64,)_128.pkl')

IPython.embed()