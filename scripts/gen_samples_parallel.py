#!/bin/env python

from spin import Model
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd

import multiprocessing as mp

import time

import IPython


def gen_samples(temp):
    geo = (16, 16)
    x, y = geo
    n_samples = 10000

    out_arr_pref = 'c' + str(n_samples) + '_g' + str(x) + str(y) + '_'
    outdir = os.path.join('.', 'signed_samples', out_arr_pref)

    x = Model()
    x.generate_system(geometry=geo, T=temp)

    start = time.time()
    x.generate_ensemble(n_samples=n_samples)
    end = time.time()
    print (temp, end-start)
    
    ec = x.ensemble.configuration
    np.save(outdir + '_T' + str(temp) + '.npy', ec)

    mag = x.ensemble.magnetization
    np.save(outdir + '_T' + str(temp) + '_mag.npy', mag)
    avg_mag = np.mean(mag)
    var_mag = np.var(mag)

    return [temp, mag, avg_mag, var_mag]

tc = 2. / np.log(1 + 2**.5)
temps = np.arange(-1, 1, .1)
temps[10] = 0
temps += tc

p = mp.Pool(4)
out = p.map(gen_samples, temps)
p.close()

mags = {}
for i in out:
    temp, mag, am, vm = i
    print (len(mag))
    mags[temp] = mag

df = pd.DataFrame(mags)
df = df.T
df['T'] = df.index / tc
dfm = pd.melt(df, id_vars=['T'])
ax = sns.violinplot(x='T', y='value', data=dfm)

plt.ylabel('<M>')

als = []
labels = ax.get_xticklabels()
for label in labels:
    als.append(label.get_text()[:5])
ax.set_xticklabels(als)

for tick in ax.get_xticklabels():
    tick.set_rotation(45)

plt.savefig('signed_samples/mag.png')

IPython.embed()
