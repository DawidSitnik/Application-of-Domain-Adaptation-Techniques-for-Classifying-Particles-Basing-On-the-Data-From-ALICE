import scipy.stats as stats
import matplotlib.pyplot as plt
from utils.config import Config
from utils.utils import load_pickle
import numpy as np


def main():
    """
    Plots comparison of attributes distributions for simulation, production and perturbed datasets.
    The plot is saved into /plots/distributions/attributes_distributions/
    """
    datasets_dict = load_pickle(f'{Config.source_fp}/pickles/datasets_dict_kaons.pkl')

    fig, axs = plt.subplots(7, figsize=(10, 10))

    for i, column in enumerate(datasets_dict['x_test_target'].drop(columns=['P'])):
        print(column)

        # perturbed dataset
        column_target = datasets_dict['x_test_target'][column]
        density = stats.gaussian_kde(column_target)
        _, x, _ = plt.hist(column_target, bins=200, histtype=u'step', density=True)
        density_x = density(x)
        axs[i].plot(density_x/np.max(density_x), label='Perturbed Dataset', alpha=0.5)

        # source dataset
        column_source = datasets_dict['x_test_source'][column]
        density = stats.gaussian_kde(column_source)
        _, x, _ = plt.hist(column_source, bins=200, histtype=u'step', density=True)
        density_x = density(x)
        axs[i].plot(density_x/np.max(density_x), label='Simulation Dataset', alpha=0.5)

        # prod dataset
        column_prod = datasets_dict['data_prod'][column]
        density = stats.gaussian_kde(column_prod)
        _, x, _ = plt.hist(column_prod, bins=200, histtype=u'step', density=True)
        density_x = density(x)
        axs[i].plot(density_x/np.max(density_x), label='Production Dataset', alpha=0.5)
        axs[i].set_title(f"{column}", fontsize=20)

    fig.tight_layout(h_pad=0.5)
    fig.savefig(
        f'{Config.source_fp}/plots/distributions/attributes_distributions/attributes_distributions.png')
    plt.close('all')


if __name__ == '__main__':
    main()
