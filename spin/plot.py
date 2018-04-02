import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=.9)
sns.set_style('white')
paper_rc = {'lines.linewidth': .8}
sns.set_context('paper', rc = paper_rc)


def plot_ensemble(model):

    """ Plot time series and marginal of
    ensemble state, energy, and magnetization """

    n_dim = model.system.n_dim
    n_spin = model.system.n_spin

    ensemble = model.ensemble
    n_samples = ensemble.n_samples

    data_lbl = ['configuration', 'energy', 'magnetization']
    data_sets = [ensemble.__dict__[i] for i in data_lbl]

    if n_dim == 2:
        configs_2d = data_sets[0]
        data_sets[0] = configs_2d.reshape(n_samples, n_spin)

    f, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios':[3, 1]})
    for i, ds_lbl in enumerate(data_lbl):
        ds = data_sets[i]

        ax = axs[i, 0]
        marg = axs[i, 1]
        if i == 0:
            sns.heatmap(ds.T, cbar=False, cmap='coolwarm', ax=ax)
            sns.barplot(ds.sum(axis=0), np.arange(n_spin), color='k',
                        orient='h', ax=marg)
            marg.set_yticks([0, n_spin])

            marg.set_yticklabels([0, n_spin])
        else:
            sns.tsplot(ds, color='k', ax=ax)
            sns.distplot(ds, color='k', vertical=True, ax=marg)

        marg.set_ylim(ax.get_ylim())
        marg.set_xticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(ds_lbl)

    ax.set_xticks([0, n_samples])
    ax.set_xticklabels([0, n_samples])
    ax.set_xlabel('MC sample')

    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(model.save_path, 'ensemble.png'), dpi=200)


def plot_rbm(model):

    """ Plot RBM output """

    rbm = model.RBM.rbm
    n_v = model.RBM.n_visible
    n_h = model.RBM.n_hidden

    f, ax = plt.subplots()
    sns.heatmap(rbm.components_, cmap='coolwarm', ax=ax)

    ax.set_xlabel('visible')
    ax.set_ylabel('hidden')
    ax.set_xticks([0, n_v])
    ax.set_xticklabels([0, n_v])
    ax.set_yticks([0, n_h])
    ax.set_yticklabels([0, n_h])

    plt.tight_layout()
    plt.savefig(os.path.join(model.save_path, 'rbm.png'), dpi=200)

