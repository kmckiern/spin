#!/bin/env python

from spin import Model
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import multiprocessing as mp

import time

import IPython


def gen_samples(temp):
    geo = (8,8)
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
temps = [tc * (.1 * i) for i in range(6,18)]

p = mp.Pool(8)
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
sns.violinplot(x='T', y='value', data=dfm)
plt.ylabel('<M>')
plt.savefig('signed_samples/mag.png')

IPython.embed()
