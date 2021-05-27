"""
Preprocess raw datasets got from Michał Kurzynka which should be placed into data/training_data dir.
Creates data_sim_preprocessed.pkl and data_prod_preprocessed.pkl in data/training_data dir.
Created data is then used for a classification.
"""
import pandas as pd
from utils.raw_data_loader import DataLoader
from utils.utils import save_pickle
import numpy as np
from scipy import stats
from utils.config import Config


def delete_outlayers(data_prod: pd.DataFrame, data_sim: pd.DataFrame, n_std=3) -> [pd.DataFrame, pd.DataFrame]:
    """
    Outlayers deletion consists of steps. In the first one we delete outlayers manually setting
    possible value ranges. In the next step, we are deleting outlayers automatically from the rest of the attributes.
    :param data_prod: production dataset
    :param data_sim: simulation dataset
    :param n_std: amount of std behind which the values are treated as outlayers
    :return: data_prod, data_sim
    """
    # step 1
    data_prod = data_prod.loc[data_prod['tpc_signal'].between(25, 150, inclusive=False)]
    data_prod = data_prod.loc[data_prod['P'].between(0, 6, inclusive=False)]
    data_prod = data_prod.loc[data_prod['its_signal'].between(25, 200, inclusive=False)]
    data_prod = data_prod.loc[data_prod['tof_signal'].between(12000, 25000, inclusive=False)]

    # step 2
    data_prod = data_prod[(np.abs(stats.zscore(data_prod)) < n_std).all(axis=1)]

    # using the same ranges in data_sim as in data_prod
    for column in data_prod.columns:
        data_sim = data_sim[data_sim[column].between(data_prod[column].min(), data_prod[column].max(), inclusive=False)]
    data_prod['pdg_code'] = None
    return data_prod, data_sim


def main():
    columns_to_drop = Config.columns_to_drop_during_preprocessing
    data_loader = DataLoader(columns_to_drop)

    data_sim = data_loader.load_file(Config.raw_data_sim_fp)
    data_prod = data_loader.load_file(Config.raw_data_prod_fp)

    columns = ['P', 'pdg_code', 'tpc_signal', 'tof_signal', 'its_signal', 'fTrackPtOnEMCal', 'fTrackEtaOnEMCal', 'fTrackPhiOnEMCal']
    data_sim = data_sim[columns].dropna()

    columns_prod = ['P', 'tpc_signal', 'tof_signal', 'its_signal', 'fTrackPtOnEMCal', 'fTrackEtaOnEMCal', 'fTrackPhiOnEMCal']
    data_prod = data_prod[columns_prod].dropna()

    data_prod, data_sim = delete_outlayers(data_prod, data_sim)

    save_pickle(data_sim, Config.preprocessed_data_sim_fp)
    save_pickle(data_prod, Config.preprocessed_data_prod_fp)


if __name__ == '__main__':
    main()
