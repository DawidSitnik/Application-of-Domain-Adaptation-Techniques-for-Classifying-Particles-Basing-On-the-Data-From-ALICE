from typing import List
import torch
import numpy as np
import pandas as pd
from .datasets import Dataset, TargetDataset
from .utils import save_pickle, load_pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from my_utils.config import Config


def compute_weights(labels_list: List) -> [float, float]:
    """
    Computes class weights used in classifiers to address the problem of imbalanced classes.
    In addition creates a pickle with classes weights.
    """
    len_labels_array = len(labels_list[labels_list == 0])
    len_0_label_frac = len_labels_array/len(labels_list[labels_list == 0])
    len_1_label_frac = len_labels_array/len(labels_list[labels_list == 1])

    return [len_0_label_frac, len_1_label_frac]


def change_labels(label_column: pd.DataFrame, main_class: int) -> pd.DataFrame:
    return label_column.apply(lambda x: 1 if x == main_class else 0)


def prepare_datasets_dict(test_size: float, seed: int, batch_size: int, class_to_predict: int) -> None:
    """
    Creates the dictionary with datasets which are used during the training and experiments.
    """

    data_prod = load_pickle(Config.preprocessed_data_prod_fp)
    data_sim = load_pickle(Config.preprocessed_data_sim_fp)

    data_prod = data_prod.loc[:, np.setdiff1d(data_prod.columns, ['pdg_code'])]
    data_sim = data_sim.sample(frac=Config.frac, random_state=0)

    Y = change_labels(data_sim['pdg_code'], class_to_predict)
    x = data_sim.loc[:, np.setdiff1d(data_sim.columns, ['pdg_code'])]

    # scaling training data
    scaler = MinMaxScaler()
    x_columns = x.columns
    x = scaler.fit_transform(x)

    # perturbing data
    x_perturbed = x.copy()
    x_perturbed[:, [1, 2, 3, 5]] = x[:, [1, 2, 3, 5]] + np.random.normal(-Config.perturbation, Config.perturbation, (len(x[:, 1]), 4))

    # additionally perturb attributes 4 and 6 which has distribution similar to normal and needs more perturbation
    x_perturbed[:, [4, 6]] = x[:, [4, 6]] + np.random.normal(-Config.additional_perturbation, Config.additional_perturbation, (len(x[:, 1]), 2))
    x_perturbed = scaler.fit_transform(x_perturbed)

    x = pd.DataFrame(x, columns=x_columns)
    x_perturbed = pd.DataFrame(x_perturbed, columns=x_columns)

    xs_train, xs_test, ys_train, ys_test = train_test_split(x, Y, test_size=test_size, random_state=seed)
    xt_train, xt_test, yt_train, yt_test = train_test_split(x_perturbed, Y, test_size=test_size, random_state=seed)

    training_weights = compute_weights(ys_train)

    train_source_dataset = Dataset(xs_train.drop(columns='P'), ys_train)
    train_source_loader = torch.utils.data.DataLoader(train_source_dataset, batch_size)

    test_source_dataset = Dataset(xs_test.drop(columns='P'), ys_test)
    test_source_loader = torch.utils.data.DataLoader(test_source_dataset, batch_size)

    x = np.bincount(train_source_loader.dataset.Y.cpu().detach().numpy())

    train_target_dataset = Dataset(xt_train.drop(columns='P'), ys_train)
    train_target_loader = torch.utils.data.DataLoader(train_target_dataset, batch_size)

    test_target_dataset = Dataset(xt_test.drop(columns='P'), ys_test)
    test_target_loader = torch.utils.data.DataLoader(test_target_dataset, batch_size)

    prod_dataset = TargetDataset(scaler.fit_transform(data_prod.drop(columns='P')))
    prod_loader = torch.utils.data.DataLoader(prod_dataset, batch_size)

    datasets_dict = {
        'x': x,
        'x_perturbed': x_perturbed,
        'x_train_source': xs_train,
        'x_test_source': xs_test,
        'y_train_source': ys_train,
        'y_test_source': ys_test,
        'x_train_target': xt_train,
        'x_test_target': xt_test,
        'y_train_target': yt_train,
        'y_test_target': yt_test,
        'train_source_dataset': train_source_dataset,
        'train_source_loader': train_source_loader,
        'test_source_dataset': test_source_dataset,
        'test_source_loader': test_source_loader,
        'train_target_dataset': train_target_dataset,
        'train_target_loader': train_target_loader,
        'test_target_dataset': test_target_dataset,
        'test_target_loader': test_target_loader,
        'prod_dataset': prod_dataset,
        'prod_loader': prod_loader,
        'data_prod': data_prod,
        'data_sim': data_sim
    }

    save_pickle(datasets_dict, f'{Config.training_data_fp}/datasets_dict_{Config.particles_dict[class_to_predict]}.pkl')
    save_pickle(training_weights, f'{Config.training_data_fp}/training_weights_{Config.particles_dict[class_to_predict]}.pkl')
