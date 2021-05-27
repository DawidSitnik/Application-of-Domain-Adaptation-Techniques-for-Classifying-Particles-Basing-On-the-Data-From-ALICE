from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from my_utils.config import Config
import matplotlib.colors as col


def plot_umap(embedding, plot_labels, plot_name, label_type):
    cmap = col.ListedColormap(["r", "b"])
    plt.figure(figsize=(30, 30))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=plot_labels, cmap=cmap, s=2, alpha=0.25)
    plt.title(plot_name.replace('_', ' '))
    plt.savefig(f'{Config.source_fp}/plots/umap/{label_type}/{plot_name}.png')


def compare_datasets_attributes(x, x_perturbed):
    for column in x:
        if column != 'P':
            print(column)
            figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
            plt.hist(x[column], bins=200, alpha=0.5, label='real')
            plt.hist(x_perturbed[column], bins=200, alpha=0.5, label='perturbed')
            plt.title(column)
            plt.legend(loc='upper right')


def draw_particles_sim(df):
    """
    plots distribution of predictions made on test simulation data vs distribution of actual labels of simulation data
    :param df: simulation dataframe
    :return: None
    """
    classes = df['prediction'].unique()
    for c in classes:
        df = df.loc[(df['P'] < 10) & (df['tpc_signal'] < 180)]

        rest_pred = df.loc[df['prediction'] != c]
        one_pred = df.loc[df['prediction'] == c]

        rest_real = df.loc[df['pdg_code'] != c]
        one_real = df.loc[df['pdg_code'] == c]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axes[0].scatter(rest_pred['P'], rest_pred['tpc_signal'], edgecolors='white', label='rest')
        axes[0].scatter(one_pred['P'], one_pred['tpc_signal'], edgecolors='white', color='orange', label=f'class {c}')
        axes[0].set_title(f'Predicted, Class {c} vs All')
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels)
        axes[0].set_xlabel('P')
        axes[0].set_ylabel('tpc_signal')

        axes[1].scatter(rest_real['P'], rest_real['tpc_signal'], edgecolors='white', label='rest')
        axes[1].scatter(one_real['P'], one_real['tpc_signal'], edgecolors='white', color='orange', label=f'class {c}')
        axes[1].set_title(f'Real, Class {c} vs All')
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels)
        axes[1].set_xlabel('P')
        axes[1].set_ylabel('tpc_signal')

        fig.tight_layout()


def draw_particles_real(df, df_sim):
    """
    plots distribution of predictions made on real data vs distribution of simulation data
    :param df: real dataset
    :param df_sim: simulation dataset
    :return: None
    """
    classes = df['prediction'].unique()
    for c in classes:
        df = df.loc[(df['P'] < 10) & (df['tpc_signal'] < 180)]
        df_sim = df_sim.loc[(df_sim['P'] < 10) & (df_sim['tpc_signal'] < 180)]

        rest_pred = df.loc[df['prediction'] != c]
        one_pred = df.loc[df['prediction'] == c]

        rest_real = df_sim.loc[df_sim['pdg_code'] != c]
        one_real = df_sim.loc[df_sim['pdg_code'] == c]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

        axes[0].scatter(rest_pred['P'], rest_pred['tpc_signal'], edgecolors='white', label='rest')
        axes[0].scatter(one_pred['P'], one_pred['tpc_signal'], edgecolors='white', color='orange', label=f'class {c}')
        axes[0].set_title(f'Predicted, Class {c} vs All')
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels)
        axes[0].set_xlabel('P')
        axes[0].set_ylabel('tpc_signal')

        axes[1].scatter(rest_real['P'], rest_real['tpc_signal'], edgecolors='white', label='rest')
        axes[1].scatter(one_real['P'], one_real['tpc_signal'], edgecolors='white', color='orange', label=f'class {c}')
        axes[1].set_title(f'Real, Class {c} vs All')
        handles, labels = axes[1].get_legend_handles_labels()
        axes[1].legend(handles, labels)
        axes[1].set_xlabel('P')
        axes[1].set_ylabel('tpc_signal')

        fig.tight_layout()


def plot_training_stats(df, metrics, model_name, particle_name, label):
    plt.figure(figsize=(15, 10))
    for metric in metrics:
        plt.plot(df[metric], label=metric)
    plt.legend(loc='upper right')
    plt.title(f'{model_name}, {particle_name}, {label}')
    plt.savefig(f'{Config.source_fp}/plots/training/{model_name}/{label}_{model_name}_{particle_name}.png')
