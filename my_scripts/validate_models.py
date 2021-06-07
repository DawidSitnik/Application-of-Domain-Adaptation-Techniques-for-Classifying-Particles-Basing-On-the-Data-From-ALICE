import matplotlib.pyplot as plt
import torch
import pandas as pd

from utils.utils import load_pickle, save_pickle, get_classifier
from utils.config import Config
from utils.validation import validate
from utils.models_dict import models_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Creates dataframe with results for each created model which
    is saved into pickles/trained_models and creates plots of
    certain training metrics.

    Scripts which needs to be run before:
    1. All the training scripts from /training_models
    """

    # get dictionary with results
    results_dict = models_validation()

    # transform into readable dataframe form
    df_columns = ['Model Name', 'Electrons', 'Pions', 'Protons', 'Kaons']
    results_df = pd.DataFrame([[model_name] + list(results_dict[model_name].values()) for model_name in results_dict],
                              columns=df_columns)
    results_df['mean'] = results_df[['Electrons', 'Kaons', 'Pions', 'Protons']].apply(
        lambda x: (x[0] + x[1] + x[2] + x[3]) / 4, axis=1)

    # sort models by mean value of all particles
    results_df.sort_values(by='mean')

    # save into pickle
    save_pickle(results_df, f'{Config.source_fp}/pickles/classification_quality/classification_quality_df.pkl')

    # creates a summary plots of certain training metric
    plot_training_stat('precision_recall_target')
    plot_training_stat('precision_recall_source')
    plot_training_stat('loss')
    plot_training_stat('trans_loss')
    plot_training_stat('loss_test')
    plot_training_stat('trans_loss_test')


def models_validation() -> dict:
    """
    Validates all the models saved in pickles/trained_models
    Creates dictionary with the results, raturns it and
    saves as pickle into /pickles/classification_quality
    """
    results_dict = {'wdgrl': {}, 'source': {}, 'cdan': {}, 'dan': {}, 'dann': {}, 'jan': {}, 'mdd': {}}

    for i, particle_name in enumerate(Config.particles_list):
        datasets_dict = load_pickle(f'{Config.training_data_fp}/datasets_dict_{particle_name}.pkl')
        for model_name in models_dict:
            print(f'validating {model_name}, {particle_name}')
            classifier = get_classifier(model_name, particle_name, models_dict)

            precision_recall_auc = validate(datasets_dict['test_target_loader'],
                                            classifier,
                                            print_classification_report=False,
                                            loud=False)
            print(precision_recall_auc)
            results_dict[model_name][particle_name] = precision_recall_auc

    save_pickle(results_dict, f'{Config.source_fp}/pickles/classification_quality/results_dict.pkl')
    print(results_dict)

    return results_dict


def plot_training_stat(training_metric: str) -> None:
    """
    Creates a summary plots of certain training metric and saves
    them into plots/training/summary as a pictures.
    """
    for particle_name in Config.particles_list:
        plt.figure(figsize=(20, 15))
        for model_name in Config.da_models_list:
            training_stats = load_pickle(
                f'{Config.source_fp}/pickles/training_stats/training_stats_df_{model_name}_{particle_name}.pkl')
            plt.plot(training_stats[training_metric], label=model_name)

        title = training_metric
        if training_metric == 'precision_recall_target':
            title = 'f1 target domain'
        if training_metric == 'precision_recall_source':
            title = 'f1 source domain'
        plt.legend(loc='upper right')
        plt.title(f'{particle_name}, {title}')
        plt.xlabel('epoch')
        plt.ylabel(title)
        plt.savefig(f'{Config.source_fp}/plots/training/summary/{training_metric}_{particle_name}.png')


if __name__ == '__main__':
    main()
