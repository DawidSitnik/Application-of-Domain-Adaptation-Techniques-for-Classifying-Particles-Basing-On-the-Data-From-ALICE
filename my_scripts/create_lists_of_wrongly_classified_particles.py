from utils.config import Config
from utils.utils import load_pickle, save_pickle, get_classifier
from utils.validation import validate
from utils.models_dict import models_dict
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_list_of_indexes_of_wrongly_classified_particles(outputs: list, targets: list) -> list:
    wrongly_classified_particles = []
    for i, x in enumerate(zip(outputs, targets)):
        if x[0] != x[1]:
            wrongly_classified_particles.append(i)
    return wrongly_classified_particles


def main():
    """
    Creates pickles with list of indexes of wrongly classified particles
    and saves them into pickles/wrongly_classified_particles

    Scripts which needs to be run before:
    1. All the training scripts from /training_models
    """

    for i, particle_name in enumerate(Config.particles_list):
        datasets_dict = load_pickle(f'{Config.training_data_fp}/datasets_dict_{particle_name}.pkl')
        for model_name in models_dict:
            print(particle_name, model_name)
            model = get_classifier(model_name, particle_name, models_dict)
            _, outputs_source, targets_source = validate(datasets_dict['test_source_loader'], model,
                                                         return_predictions=True)
            _, outputs_target, targets_target = validate(datasets_dict['test_target_loader'], model,
                                                         return_predictions=True)

            wrongly_classified_particles_source = get_list_of_indexes_of_wrongly_classified_particles(outputs_source, targets_source)
            wrongly_classified_particles_target = get_list_of_indexes_of_wrongly_classified_particles(outputs_target, targets_target)

            save_pickle(wrongly_classified_particles_source,
                        f'{Config.source_fp}/pickles/wrongly_classified_particles/{particle_name}_{model_name}_wrongly_classified_particles_source.pkl')

            save_pickle(wrongly_classified_particles_target,
                        f'{Config.source_fp}/pickles/wrongly_classified_particles/{particle_name}_{model_name}_wrongly_classified_particles_target.pkl')


if __name__ == '__main__':
    main()
