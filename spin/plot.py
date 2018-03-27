import matplotlib.pyplot as plt
import seaborn as sns

def plot_ensemble(model):
    ensemble = model.ensemble
    config = ensemble.configuration
    magnetization = ensemble.magnetization
    energies = ensemble.energies
    save_path = model.save_path

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    sns.heatmap(config.T, cbar=False, cmap='coolwarm', ax=ax1)
    sns.tsplot(energies, color='k', ax=ax2)
    sns.tsplot(magnetization, color='k', ax=ax3)

    ax1.set_ylabel('particle')
    ax1.set_yticks([])
    ax2.set_ylabel('energy')
    ax3.set_ylabel('magnetization')
    ax3.set_xlabel('MC step')
    ax3.set_xticks([])
    plt.tight_layout()
    plt.savefig(save_path + '_ensemble.png', dpi=200)
