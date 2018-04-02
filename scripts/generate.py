#!/bin/env python

from spin import Model
import os
import numpy as np


def gen_samples(temp, geo, n_samples):

    lbl = '_'.join([str(temp)[:4].replace('.', 'p'),
                    str(geo).replace(',','x')[1:-1], str(n_samples)])
    lbl = lbl.replace(' ', '')
    sample_dir = os.path.join('..', 'samples', lbl)

    x = Model(save_path=sample_dir)
    x.generate_system(geometry=geo, T=temp)

    x.generate_ensemble(n_samples=n_samples)
    x.describe('ensemble', plot_component=True)

    x.generate_RBM(optimize=True)
    x.describe('RBM', plot_component=True)

    x.save_model()


tc = 2. / np.log(1 + 2**.5)
temps = np.arange(-1, 1, .1)
temps[10] = 0
temps += tc

gen_samples(temps[-1], geo=(4,4), n_samples=50)
