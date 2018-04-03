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



def describe(self):
    label = ' ,'.join(['batch_size ' + str(self.batch_size),
                      'learning_rate ' + str(self.learning_rate)])
    return label


"""
def plot_train(model):

    In[24]: len(x.VAE.vae.train_log['encoded'])
    Out[24]: 10

    In[25]: x.VAE.vae.train_log['encoded'][0].shape
    Out[25]: (4, 128)

    In[3]: x.VAE.vae.train_log.keys()
    Out[3]: dict_keys(['error', 'encoded', 'decoded', 'reference'])

    In[4]: x.VAE.vae.train_log['reference']
    Out[4]:
    array([[1., 1., -1., ..., 1., 1., 1.],
           [-1., -1., -1., ..., -1., -1., -1.],
           [1., -1., -1., ..., 1., 1., 1.],
           [1., 1., 1., ..., 1., 1., -1.]], dtype=float32)

    In[5]: x.VAE.vae.train_log['reference'].shape
    Out[5]: (4, 256)

    In[6]: x.VAE.vae.train_log['decoded'][-1].shape
    Out[6]: (4, 256)

    if watch_fit:
        NVIEW = 5
        f, axs = plt.subplots(2, NVIEW, sharex=True, sharey=True,
                              figsize=(10, 4))
        for ref in range(NVIEW):
            ax = axs[0][ref]
            data_ref = training_data[ref]
            if self.n_dim == 1:
                s_1d = data_ref.size
                ref_plt = data_ref.reshape(1, s_1d)
            else:
                s_2d = data_ref.shape
                ref_plt = data_ref
            sns.heatmap(ref_plt, ax=ax,
                        cbar=False, cmap='coolwarm', square=True)
            ax.set_xticks([])
            ax.set_yticks([])

    if watch_fit and (global_step % write_freq == 0):
        for ref in range(NVIEW):
            ax = axs[1][ref]
            data_ref = decoded.data.numpy()[ref]
            if self.n_dim == 2:
                decode_plt = data_ref.reshape(s_2d)
            else:
                decode_plt = data_ref.reshape(1, s_1d)
            sns.heatmap(decode_plt, ax=ax, cbar=False,
                        cmap='coolwarm', square=True)
            ax.set_xticks([])
            ax.set_yticks([])
            if ref == 0:
                ax.set_title('epoch: ' + str(epoch) + ', reconstruction error: ' + str(error.data[0])[:6])
        plt.savefig(os.path.join(save_dir, 's' + str(global_step).zfill(8) + '.png'))

        sd = os.path.join(self.save_path, str(cndx + 4))
        if not os.path.exists(sd):
            os.mkdir(sd)
"""
