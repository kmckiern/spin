#!/bin/env python

from spin import Model
import IPython


# load model object
x = Model()
x.load_model('../samples/3p16_64x_512/model.pkl')

IPython.embed()