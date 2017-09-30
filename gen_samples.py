#!/bin/env python

from spin import Model
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

geo = (16, 16)
nsamp = 1000
outdir = os.path.join('.', 'samples', 'c1000_g1616_')

def gen_samples(temp, geo=geo, n_samples=nsamp):

    x = Model()
    x.generate_system(geometry=geo, T=temp)
    x.generate_ensemble(n_samples=n_samples)
    
    ec = x.ensemble.configuration
    np.save(outdir + '_T' + str(temp) + '.npy', ec)

    mag = x.ensemble.magnetization
    avg_mag = np.mean(mag)
    var_mag = np.var(mag)

    return [avg_mag, var_mag]

temps = np.arange(0.5, 3.5, .25)
m_avgs = []
m_vars = []
for t in temps:
    print (t)
    a, v = gen_samples(t)
    m_avgs.append(a)
    m_vars.append(v)

plt.figure()
plt.errorbar(temps, m_avgs, yerr=m_vars, fmt='--o')
plt.ylabel('<M>')
plt.xlabel('T')
plt.savefig(outdir + 'mag_T.png')
