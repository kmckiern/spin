from spin.model import Model
from spin.plot import plot_ensemble, plot_rbm
import os
import numpy as np


def gen_samples(temp, geo, n_samples):

    lbl = '_'.join([str(temp)[:4].replace('.', 'p'),
                    str(geo).replace(',', 'x')[1:-1], str(n_samples)])
    lbl = lbl.replace(' ', '')
    sample_dir = os.path.join('..', 'samples', lbl)

    x = Model(geometry=geo, T=temp, save_path=sample_dir)

    model_file = os.path.join(sample_dir, 'model.pkl')
    if not os.path.exists(model_file):
        x.generate_ensemble(n_samples, autocorrelation_threshold=.5)
        x.generate_rbm()
        x.save_model()
    else:
        x.load_model(model_file)

    plot_ensemble(x)
    plot_rbm(x)


tc = 2. / np.log(1 + 2**.5)
temps = np.arange(-1, 1, .1)
temps[10] = 0
temps += tc

gen_samples(temps[-1], geo=(10, 10), n_samples=100)
