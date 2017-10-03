#!/bin/env python

import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import multiprocessing as mp

import IPython


def load_samples(temp):
    geo = (8,8)
    x, y = geo
    n_samples = 5000

    out_arr_pref = 'c' + str(n_samples) + '_g' + str(x) + str(y) + '_'
    outdir = os.path.join('.', out_arr_pref)
    mag = np.load(outdir + '_T' + str(temp) + '_mag.npy')
    avg_mag = np.mean(mag)
    var_mag = np.var(mag)

    return [temp, mag, avg_mag, var_mag]

temps = np.arange(1.5, 3.75, .25)[::-1]

p = mp.Pool(len(temps))
out = p.map(load_samples, temps)
p.close()

mags = {}
ts = []
vs = []
avs = []
for i in out:
    temp, mag, am, vm = i
    mags[temp] = mag
    ts.append(temp)
    avs.append(am)
    vs.append(vm)

df = pd.DataFrame(mags)
df = df.T
df['T'] = df.index
dfm = pd.melt(df, id_vars=['T'])
sns.violinplot(x='T', y='value', data=dfm)
plt.savefig('mag_violin.png')

plt.clf()
plt.figure()
plt.errorbar(ts, avs, yerr=vs, fmt='--o')
plt.savefig('mag_yerr.png')

IPython.embed()
