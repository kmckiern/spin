#!/bin/env python

from spin import Model
import IPython


# load model object
x = Model()
x.load_model('../samples/3p16_4x4_50/model.pkl')

IPython.embed()
