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

    ensemble = model.ensemble
    n_samples = ensemble.n_samples
    n_particles = ensemble.geometry[0]

    data = ['configuration', 'energies', 'magnetization']

    f, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios':[3, 1]})
    for i, ds_lbl in enumerate(data):
        ds = ensemble.__dict__[ds_lbl]

        ax = axs[i, 0]
        marg = axs[i, 1]
        if i == 0:
            sns.heatmap(ds.T, cbar=False, cmap='coolwarm', ax=ax)
            sns.barplot(ds.sum(axis=0), np.arange(n_particles), color='k',
                        orient='h', ax=marg)
            marg.set_yticks([0, n_particles])

            marg.set_yticklabels([0, n_particles])
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

    rbm = model.network.rbm
    n_v = model.network.n_neurons
    n_h = model.network.n_hidden
    v_rs = int(np.ceil(n_v ** .5))
    h_rs = int(np.ceil(n_h ** .5))

    rbm_c = rbm.components_
    nc = len(rbm_c)
    c_max = rbm_c.max()
    c_min = rbm_c.min()

    f, ax = plt.subplots()
    sns.heatmap(rbm.components_, cbar=False, cmap='coolwarm', ax=ax)

    ax.set_xlabel('visible')
    ax.set_ylabel('hidden')
    ax.set_xticks([0, n_v])
    ax.set_xticklabels([0, n_v])
    ax.set_yticks([0, n_h])
    ax.set_yticklabels([0, n_h])

    plt.tight_layout()
    plt.savefig(os.path.join(model.save_path, 'rbm.png'), dpi=200)

    f, axs = plt.subplots(h_rs, h_rs)
    faxs = axs.flat
    for i, c_ax in enumerate(faxs):
        if i < nc:
            comp = rbm_c[i]
            data = comp.reshape((v_rs, v_rs))
            sns.heatmap(data, cbar=False, cmap='coolwarm', ax=c_ax,
                        vmin=c_min, vmax=c_max)
        else:
            sns.despine(left=True, bottom=True)
        c_ax.set_yticks([])
        c_ax.set_xticks([])

    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(model.save_path, 'activations.png'), dpi=200)
