#!/bin/env python

from spin import Model
import os
import numpy as np


def gen_samples(temp):
    geo = (64,)
    n_samples = 128
    
    lbl = sample_dir
    lbl += '_'.join(['T' + str(temp)[:4], str(geo), str(n_samples)])

    x = Model()
    x.generate_system(geometry=geo, T=temp)

    x.generate_ensemble(n_samples=n_samples)
    x.describe_ensemble()

    x.generate_RBM()
    
    x.save_model(lbl + '.pkl')

    return None

sample_dir = os.path.join('..', 'samples/unit/')

tc = 2. / np.log(1 + 2**.5)
temps = np.arange(-1, 1, .1)
temps[10] = 0
temps += tc

gen_samples(temps[-1])
