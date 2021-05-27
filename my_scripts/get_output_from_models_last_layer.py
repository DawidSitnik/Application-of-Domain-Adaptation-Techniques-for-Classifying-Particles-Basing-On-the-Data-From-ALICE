from utils.config import Config
from utils.utils import load_pickle, save_pickle, get_classifier, get_model_features
import torch
from utils.models_dict import models_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Gets output vector from feature extractor of each model and saves it into /pickles/last_layer_output/

    Scripts which needs to be run before:
    1. All the training scripts from /training_models
    """
    print('Getting outputs from models last layer.')
    for i, particle_name in enumerate(Config.particles_list):
        datasets_dict = load_pickle(f'{Config.training_data_fp}/datasets_dict_{particle_name}.pkl')
        for model_name in models_dict:
            print(f'getting output for {particle_name}, {model_name}')
            model = get_classifier(model_name, particle_name, models_dict)
            test_source_features = get_model_features(model, datasets_dict['test_source_loader'])
            test_target_features = get_model_features(model, datasets_dict['test_target_loader'])

            save_pickle(test_source_features,
                        f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_source_last_layer_output.pkl')
            save_pickle(test_target_features,
                        f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_target_last_layer_output.pkl')


if __name__ == '__main__':
    main()
