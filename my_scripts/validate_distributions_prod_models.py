import torch
import pandas as pd
from utils.utils import load_pickle, save_pickle, get_classifier
from utils.config import Config
from utils.validation_prod import validate
from utils.models_dict import models_dict
import matplotlib.pyplot as plt
import seaborn as sns
from utils.datasets import Dataset, TargetDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = load_pickle(f'{Config.training_data_michal_fp}/data_scaler.pkl')


def main():
    """

    """
    models_validation()


def models_validation() -> dict:
    """

    """
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for i, particle_name in enumerate(Config.particles_list):
        datasets_dict = load_pickle(f'{Config.training_data_michal_fp}/datasets_dict_{i}.pkl')
        plot_original_p_vs_tpc(axes[0], datasets_dict, particle_name)

        for model_name in models_dict:
            if model_name not in ['source', 'dann']:
                continue
            print(f'validating {model_name}, {particle_name}')
            classifier = get_classifier(model_name, particle_name, models_dict)
            pdg_code = validate(datasets_dict['prod_loader'], classifier)
            columns = datasets_dict['data_prod'].columns
            prod_df = pd.DataFrame(scaler.inverse_transform(datasets_dict['data_prod']), columns=columns)
            prod_df['pdg_code'] = pdg_code
            plot_p_vs_tpc(prod_df.query('pdg_code == 1'), model_name, particle_name)


def plot_original_p_vs_tpc(axes, datasets_dict, particle_name):
    x_train_source = datasets_dict['x_train_source']
    columns = x_train_source.columns
    x_train_source = pd.DataFrame(scaler.inverse_transform(x_train_source), columns=columns)
    x_train_source['pdg_code'] = datasets_dict['y_train_source']
    plot_p_vs_tpc(axes, x_train_source.query('pdg_code == 1'), 'original', particle_name)

def plot_p_vs_tpc(axes, data_frame, model_name, particle_name, fig_size=(15,8)):
    fig = plt.figure(figsize=fig_size, dpi=80)
    sns.scatterplot(x="P", y="tpc_signal", data=data_frame, palette="deep")
    plt.xscale('log')
    plt.savefig(f'{Config.source_fp}/plots/p_vs_tpc/{particle_name}_{model_name}_p_vs_tpc.png')


if __name__ == '__main__':
    main()
