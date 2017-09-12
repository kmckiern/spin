#!/bin/env python

import numpy as np
import glob
import natsort
import matplotlib.pyplot as plt
import seaborn as sns
import IPython

ensemble_files = natsort.natsorted(glob.glob('*npy'))

temps = []
avgs = []
stds = []
for f in ensemble_files:
    temp = float(f.split('_')[0].split('T')[-1])

    configurations = np.load(f)
    mag = configurations.sum(1).sum(1) / configurations[0].size
    avg = mag.mean()
    std = mag.std()

    temps.append(temp)
    avgs.append(avg)
    stds.append(std)

plt.figure()
plt.errorbar(temps, avgs, yerr=stds, fmt='--o')
plt.ylabel('<M>')
plt.xlabel('T')
plt.savefig('mag_T.png')
