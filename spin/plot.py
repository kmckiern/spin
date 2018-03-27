import matplotlib.pyplot as plt
import seaborn as sns

def plot_ensemble(model):

    """ Plot time series of ensemble state, energy, and magnetization """

    ensemble = model.ensemble

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    sns.heatmap(ensemble.configuration.T, cbar=False, cmap='coolwarm', ax=ax1)
    sns.tsplot(ensemble.energies, color='k', ax=ax2)
    sns.tsplot(ensemble.magnetization, color='k', ax=ax3)

    ax1.set_ylabel('particle')
    ax2.set_ylabel('energy')
    ax3.set_ylabel('magnetization')
    ax3.set_xlabel('MC step')

    ax1.set_yticks([])
    ax3.set_xticks([])

    plt.tight_layout()
    plt.savefig(model.save_path + '_ensemble.png', dpi=200)
