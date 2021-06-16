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

    for i, particle_name in enumerate(Config.particles_list):
        datasets_dict = load_pickle(f'{Config.training_data_michal_fp}/datasets_dict_{i}.pkl')
        for model_name in models_dict:
            if model_name not in ['source', 'wdgrl', 'dann', 'wdgrl_michal']:
                continue
            classifier = get_classifier(model_name, particle_name, models_dict)
            pdg_code = validate(datasets_dict['prod_loader'], classifier)
            columns = datasets_dict['data_prod'].columns
            # prod_df = datasets_dict['data_prod']
            prod_df = pd.DataFrame(scaler.inverse_transform(datasets_dict['data_prod']), columns=columns)
            prod_df['pdg_code'] = pdg_code
            # prod_df['pdg_code'] = validate(datasets_dict['test_source_loader'], classifier)
            avg_gauss_prob, avg_model_prob, avg_efficiency, avg_accuracy = estimate_classification_quality(prod_df,
                                                                                                           particles_translation[particle_name])
            print(f'{model_name}, {particle_name}, efficiency: {avg_efficiency}, accuracy: {avg_accuracy}')
    return results_dict


if __name__ == '__main__':
    main()
