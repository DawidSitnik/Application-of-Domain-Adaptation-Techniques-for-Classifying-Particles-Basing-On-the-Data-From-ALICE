import os

import torch
import pandas as pd
from utils.utils import load_pickle, save_pickle, get_classifier
from utils.config import Config
from utils.validation_prod import validate
from utils.models_dict import models_dict
import matplotlib.pyplot as plt
import seaborn as sns
from utils.datasets import Dataset, TargetDataset
import numpy as np
import PIL
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = load_pickle(f'{Config.training_data_michal_fp}/data_scaler.pkl')


def main():
    """

    """
    models_validation()


def models_validation() -> dict:
    """

    """
    plots_dir_path = '../data/plots/p_vs_tpc/'

    # Config.particles_list = ['pions']

    for i, particle_name in enumerate(Config.particles_list):
        datasets_dict = load_pickle(f'{Config.training_data_michal_fp}/datasets_dict_{i}.pkl')
        plot_original_p_vs_tpc(datasets_dict, particle_name)

        for model_name in models_dict:
            if model_name not in ['source', 'wdgrl', 'dann']:
                continue
            print(f'validating {model_name}, {particle_name}')
            classifier = get_classifier(model_name, particle_name, models_dict)
            pdg_code = validate(datasets_dict['prod_loader'], classifier)
            columns = datasets_dict['data_prod'].columns
            prod_df = pd.DataFrame(scaler.inverse_transform(datasets_dict['data_prod']), columns=columns)
            prod_df['pdg_code'] = pdg_code
            prod_df = prod_df.sample(len(datasets_dict['x_train_source']))  # for better comparison
            plot_p_vs_tpc(prod_df.query('pdg_code == 1'), model_name, particle_name)

        merge_plots(particle_name, plots_dir_path)

    delete_unused_files(plots_dir_path)


def plot_original_p_vs_tpc(datasets_dict, particle_name):
    x_train_source = datasets_dict['x_train_source']
    columns = x_train_source.columns
    x_train_source = pd.DataFrame(scaler.inverse_transform(x_train_source), columns=columns)
    x_train_source['pdg_code'] = datasets_dict['y_train_source']
    plot_p_vs_tpc(x_train_source.query('pdg_code == 1'), 'original', particle_name)


def plot_p_vs_tpc(data_frame, model_name, particle_name):
    sns.scatterplot(x="P", y="tpc_signal", data=data_frame, palette="deep")
    plt.xscale('log')
    plt.title(f'{particle_name}, {model_name}')
    plt.savefig(f'{Config.source_fp}/plots/p_vs_tpc/{particle_name}_{model_name}_p_vs_tpc.png')
    plt.clf()


def merge_plots(particle_name, plots_dir_path):
    plot_files = [x for x in os.listdir(plots_dir_path) if particle_name in x and 'merged' not in x]
    images = [PIL.Image.open(plots_dir_path+i) for i in plot_files]

    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]
    imgs_comb = np.hstack([np.asarray(i.resize(min_shape)) for i in images])

    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save(f'{Config.source_fp}/plots/p_vs_tpc/{particle_name}_p_vs_tpc_merged.png')

    imgs_comb = np.vstack([np.asarray(i.resize(min_shape)) for i in images])
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save(f'{Config.source_fp}/plots/p_vs_tpc/{particle_name}_p_vs_tpc_vertical_merged.png')


def delete_unused_files(plots_dir_path):
    files_to_delete = [x for x in os.listdir(plots_dir_path) if 'merged' not in x]
    for file in files_to_delete:
        os.remove(plots_dir_path+file)


if __name__ == '__main__':
    main()
