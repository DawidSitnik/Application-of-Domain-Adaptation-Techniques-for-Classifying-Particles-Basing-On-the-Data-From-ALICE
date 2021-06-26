import torch
import pandas as pd
from utils.datasets import Dataset, TargetDataset
from utils.utils import load_pickle, save_pickle, get_classifier
from utils.config import Config
from utils.validation_prod import validate, estimate_classification_quality
from utils.models_dict import models_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = load_pickle(f'{Config.training_data_michal_fp}/data_scaler.pkl')


def main():
    """

    """

    # get dictionary with results
    models_validation()


def models_validation() -> dict:
    """
    ogarnac dlaczego w colabie i lokalnie dostaje inne wyniki, moze sprawdzic jeszcze w jupyterze?
    """
    # results_dict = {'wdgrl': {}, 'source': {}, 'cdan': {}, 'dan': {}, 'dann': {}, 'jan': {}, 'mdd': {}}
    results_dict = {'source': {}, 'wdgrl': {}, 'dann': {}, 'wdgrl_michal':{}}

    # pions, kaons, electrons, protons
    particles_translation = {
        'pions': 0,
        'kaons': 1,
        'electrons': 2,
        'protons': 3
    }

    Config.particles_list = ['protons']

    for i, particle_name in enumerate(Config.particles_list):
        datasets_dict = load_pickle(f'{Config.training_data_michal_fp}/datasets_dict_{i}.pkl')
        for model_name in models_dict:
            if model_name not in ['source', 'wdgrl', 'wdgrl_michal']:
            # if model_name not in ['source']:
                continue
            classifier = get_classifier(model_name, particle_name, models_dict)
            pdg_code = validate(datasets_dict['prod_loader'], classifier)
            columns = datasets_dict['data_prod'].columns
            # prod_df = datasets_dict['data_prod']
            prod_df = pd.DataFrame(scaler.inverse_transform(datasets_dict['data_prod']), columns=columns)
            prod_df['pdg_code'] = pdg_code
            if model_name == 'wdgrl_michal':
                x1 = pdg_code.count(0)
                x2 = pdg_code.count(1)
                x3=0
            save_pickle(prod_df, '/home/dawid/Documents/pid_mkurzynka/src/dataprep/my_prod_protons.pkl')
            # prod_df['pdg_code'] = validate(datasets_dict['test_source_loader'], classifier)
            avg_gauss_prob, avg_model_prob, avg_efficiency, avg_accuracy, avg_f1 = estimate_classification_quality(prod_df,
                                                                                                           particles_translation[particle_name], model_name, particle_name)
            print(f'{model_name}, {particle_name}, efficiency: {avg_efficiency}, accuracy: {avg_accuracy}, f1: {avg_f1}')
    return results_dict


if __name__ == '__main__':
    main()
