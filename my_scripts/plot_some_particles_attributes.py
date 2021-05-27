import pandas as pd

from utils.utils import load_pickle
from utils.config import Config
import scipy.stats as stats
import matplotlib.pyplot as plt


def main():
    """
    Plots distributions of attributes for unadapted and wrongly classified particles.
    Plots are saved into plots/unadapted_particles/ and plots/wrongly_classified_particles/

    Note:
        - basing on Michał Kurzynka experiments the most important
          attributes are probably: tpc_signal and its_signal

    Scripts which needs to be run before:
    1. All the training scripts from /training_models
    2. create_lists_of_wrongly_classified_particles.py
    """

    for i, particle_name in enumerate(Config.particles_list):

        datasets_dict = load_pickle(f'{Config.training_data_fp}/datasets_dict_{particle_name}.pkl')
        wrongly_classified_particles_source = {}
        wrongly_classified_particles_target = {}
        unadapted_particles_source = {}
        unadapted_particles_target = {}

        for model_name in Config.da_models_list:
            wrongly_classified_particles_source[model_name] = load_pickle(
                f'{Config.source_fp}/pickles/wrongly_classified_particles/{particle_name}_{model_name}_wrongly_classified_particles_source.pkl')
            wrongly_classified_particles_target[model_name] = load_pickle(
                f'{Config.source_fp}/pickles/wrongly_classified_particles/{particle_name}_{model_name}_wrongly_classified_particles_target.pkl')

            unadapted_particles_source[model_name] = load_pickle(
                f'{Config.source_fp}/pickles/unadapted_particles/{particle_name}_{model_name}_unadapted_particles_source.pkl')
            unadapted_particles_target[model_name] = load_pickle(
                f'{Config.source_fp}/pickles/unadapted_particles/{particle_name}_{model_name}_unadapted_particles_target.pkl')

        print('plotting distribution of wrongly classified particles.')
        plot_attributes_comparison(
            wrongly_classified_particles_source,
            datasets_dict['x_test_source'].drop(columns=['P']),
            wrongly_classified_particles_target,
            datasets_dict['x_test_target'].drop(columns=['P']),
            particle_name, particles_type='wrongly_classified_particles')

        print('plotting distribution of unadapted particles')
        plot_attributes_comparison(
            unadapted_particles_source,
            datasets_dict['x_test_source'].drop(columns=['P']),
            unadapted_particles_target,
            datasets_dict['x_test_target'].drop(columns=['P']),
            particle_name, particles_type='unadapted_particles')


def plot_attributes_comparison(particles_to_compare_source: pd.DataFrame, x_test_source: pd.DataFrame,
                               particles_to_compare_target: pd.DataFrame, x_test_target: pd.DataFrame,
                               particles_name: str, particles_type: str) -> None:
    """
    Plots distribution of wrongly classified / unadapted particles with the distribution of all the particles.
    The comparisons are made separately for source and target domain.
    Plots are saved into /plots/distributions/wrongly_classified or /plots/distributions/unadapted_particles

    Parameters:
        - particles_to_compare_source - attributes of wrongly classified / unadapted particles from source domain
        - x_test_source - attributes of all particles from testing source domain
        - particles_to_compare_target - attributes of wrongly classified / unadapted particles from target domain
        - x_test_target - attributes of all particles from testing target domain
        - particles_type - type of particles to plot. Can be wrongly_classified_particles or unadapted_particles
    """
    if particles_type == 'wrongly_classified_particles':
        title = 'Wrongly Classified Particles'
    else:
        title = 'Unadapted Particles'

    for column in x_test_source:
        fig, axs = plt.subplots(4)
        plt.figure(num=None, figsize=(20, 45), dpi=80, facecolor='w', edgecolor='k')

        for model_name in particles_to_compare_source:
            try:
                if particles_type == 'wrongly_classified_particles':
                    values_to_plot = x_test_source.iloc[particles_to_compare_source[model_name]][column]
                else:
                    values_to_plot = particles_to_compare_source[model_name][column]
                density = stats.gaussian_kde(values_to_plot)
                n, x, _ = plt.hist(values_to_plot, bins=200, histtype=u'step', density=False)
                axs[0].plot(x, density(x), label=model_name)
                axs[0].legend(loc="upper right", prop={'size': 6})
                axs[0].set_title(f"Density Function of {title} in Source Dataset ({column}, {model_name})", fontsize=8)
            except:
                pass

        for model_name in particles_to_compare_target:
            try:
                if particles_type == 'wrongly_classified_particles':
                    values_to_plot = x_test_target.iloc[particles_to_compare_target[model_name]][column]
                else:
                    values_to_plot = particles_to_compare_target[model_name][column]

                density = stats.gaussian_kde(values_to_plot)
                n, x, _ = plt.hist(values_to_plot, bins=200, histtype=u'step', density=True)
                axs[1].plot(x, density(x), label=model_name)
                axs[1].legend(loc="upper right", prop={'size': 6})
                axs[1].set_title(f"Density Function of {title} in Perturbed Dataset ({column})", fontsize=8)

                axs[2].hist(values_to_plot, bins=10, histtype=u'step', density=False, label=model_name)
                axs[2].legend(loc="upper right", prop={'size': 6})
                axs[2].set_title(f"Histogram of {title} in Perturbed Dataset ({column})", fontsize=8)
            except:
                pass

        values_to_plot = x_test_target[column]
        density = stats.gaussian_kde(values_to_plot)
        n, x, _ = plt.hist(values_to_plot, bins=200, histtype=u'step', density=True)
        axs[3].plot(x, density(x), label='Perturbed Dataset')
        values_to_plot = x_test_source[column]
        density = stats.gaussian_kde(values_to_plot)
        n, x, _ = plt.hist(values_to_plot, bins=200, histtype=u'step', density=True)
        axs[3].plot(x, density(x), label='Source Dataset')
        axs[3].legend(loc="upper right",prop={'size': 6})
        axs[3].set_title(f"Density Function of Attribute {column} in Perturbed and Source Dataset.", fontsize=8)

        fig.tight_layout(h_pad=0.5)
        fig.savefig(f'{Config.source_fp}/plots/distributions/{particles_type}/{particles_name}_{column}.pdf')
        plt.close('all')


if __name__ == '__main__':
    main()
