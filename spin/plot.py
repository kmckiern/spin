import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=0.9)
sns.set_style("white")
paper_rc = {"lines.linewidth": 0.8}
sns.set_context("paper", rc=paper_rc)


def plot_ensemble(model):

    """ Plot time series and marginal of
    ensemble state, energy, and magnetization """

    n_dim = model.configuration.ndim
    n_spin = model.configuration.size

    ensemble = model.ensemble
    n_samples = len(ensemble)

    data_sets = [model.ensemble, model.energies, model.magnetization]
    ds_lbl = ["configurations", "energy", "magnetization"]

    if n_dim == 2:
        configs_2d = data_sets[0]
        data_sets[0] = configs_2d.reshape(n_samples, n_spin)

    f, axs = plt.subplots(3, 2, gridspec_kw={"width_ratios": [3, 1]})
    for i, ds in enumerate(data_sets):
        ax = axs[i, 0]
        marg = axs[i, 1]

        if i == 0:
            sns.heatmap(ds.T, cbar=False, cmap="coolwarm", ax=ax)
            sns.barplot(
                ds.sum(axis=0), np.arange(n_spin), color="k", orient="h", ax=marg
            )
            marg.set_yticks([0, n_spin])

            marg.set_yticklabels([0, n_spin])
        else:
            sns.tsplot(ds, color="k", ax=ax)
            sns.distplot(ds, color="k", vertical=True, ax=marg)

        marg.set_ylim(ax.get_ylim())
        marg.set_xticks([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(ds_lbl[i])

    ax.set_xticks([0, n_samples])
    ax.set_xticklabels([0, n_samples])
    ax.set_xlabel("MC sample")

    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(model.save_path, "ensemble.png"), dpi=200)


def plot_rbm(model):
    """ Plot RBM output """

    rbm = model.RBM.rbm
    n_v = model.RBM.n_visible
    n_h = model.RBM.n_hidden

    f, ax = plt.subplots()
    sns.heatmap(rbm.components_, cmap="coolwarm", ax=ax)

    ax.set_xlabel("visible")
    ax.set_ylabel("hidden")
    ax.set_xticks([0, n_v])
    ax.set_xticklabels([0, n_v])
    ax.set_yticks([0, n_h])
    ax.set_yticklabels([0, n_h])

    plt.tight_layout()
    plt.savefig(os.path.join(model.save_path, "rbm.png"), dpi=200)


def plot_train(model, n_type="VAE"):
    """ Plot error aafo epoch, conditioned on a set of hyperparameters `"""

    f, ax = plt.subplots()

    trained_networks = model.__dict__[n_type].trained_models

    scores = sorted(list(trained_networks.keys()))
    for score in scores:
        trained_network = trained_networks[score]
        lbl = (trained_network.learning_rate, trained_network.batch_size)

        epochs = np.arange(trained_network.n_iter)
        errors = trained_network.train_log["error"]
        ax.plot(epochs, errors, "o-", label=lbl)

    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(model.save_path, n_type + "_opt.png"), dpi=200)


def plot_reconstruction(model, vae_model):
    geometry = model.geometry

    train_log = vae_model.scores["train_log"]
    n_log = train_log["n_log"]

    movie_dir = os.path.join(model.save_path, "train_movie")
    if not os.path.exists(movie_dir):
        os.makedirs(movie_dir)

    f, axs = plt.subplots(2, n_log, figsize=(2 * n_log, 5))
    for ref in range(n_log):
        ax = axs[0][ref]
        sns.heatmap(
            train_log["references"][ref].reshape(geometry),
            ax=ax,
            cbar=False,
            square=True,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for iter in range(len(train_log["reconstructed"])):
        for ref in range(n_log):
            ax = axs[1][ref]
            sns.heatmap(
                train_log["reconstructed"][iter][ref].reshape(geometry),
                ax=ax,
                cbar=False,
                square=True,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        axs[1][0].set_title(
            "epoch: {}\nloss: {:05f}".format(iter, train_log["error"][iter]), loc="left"
        )

        plt.tight_layout()
        plt.savefig(os.path.join(movie_dir, "i" + str(iter).zfill(8) + ".png"))
