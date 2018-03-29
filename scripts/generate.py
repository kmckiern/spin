#!/bin/env python

from spin import Model
import os
import numpy as np


def gen_samples(temp, geo=(64,), n_samples=512):

    lbl = '_'.join([str(temp)[:4].replace('.', 'p'),
                    str(geo).replace(',','x')[1:-1], str(n_samples)])
    sample_dir = os.path.join('..', 'samples', lbl)

    x = Model()
    x.generate_system(geometry=geo, T=temp, save_path=sample_dir)
    x.generate_ensemble(n_samples=n_samples)
    x.describe_ensemble()
    x.generate_RBM()
    x.save_model()


tc = 2. / np.log(1 + 2**.5)
temps = np.arange(-1, 1, .1)
temps[10] = 0
temps += tc

gen_samples(temps[-1])
