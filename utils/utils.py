import os
import pandas as pd
import torch
import pickle
import numpy as np
from domain_adaptation.modules.classifier import Classifier as SourceClassifier
from utils.models import Net
from utils.config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def does_file_exist(file_name: str) -> bool:
    return os.path.isfile(file_name)


def save_pickle(data, fp):
    with open(fp, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(fp):
    with open(fp, 'rb') as handle:
        b = pickle.load(handle)
    return b


def pred_data_loader_dann(dl, model):
    classes_list = []
    domains_list = []
    for data in dl:
        inputs, _ = data
        inputs = inputs.cuda()
        classes, domains = model(inputs)
        _, classes = torch.max(classes.data, 1)
        _, domains = torch.max(domains.data, 1)
        classes_list += list(classes.cpu().numpy())
        domains_list += list(domains.cpu().numpy())

    return classes_list, domains_list


def get_model_features(classifier, loader):
    features = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            _, f = classifier(x)
            features.extend(f.cpu().numpy().tolist())
    features = np.array(features)

    return features


def get_dataset(source_features, target_features):
    df_source = pd.DataFrame(source_features)
    df_source['domain'] = 0
    df_target = pd.DataFrame(target_features)
    df_target['domain'] = 1

    return df_source.append(df_target, ignore_index=True)

from utils.wdgrl_michal import Network

def get_classifier(model_name, particle_name, models_dict):
    backbone = Net(Config.n_features)
    source_classifier = SourceClassifier(backbone, Config.n_features, classifier_type='source').to(device)

    if model_name == 'wdgrl':
        source_classifier = source_classifier.backbone
        source_classifier.load_state_dict(
            torch.load(f'{Config.wdgrl_model_fp}_{particle_name}.pt',
                       map_location=torch.device(device)))
        return source_classifier

    elif model_name == 'source':
        source_classifier.load_state_dict(
            torch.load(f'{Config.source_model_fp}_{particle_name}.pt',
                       map_location=torch.device(device)))
        return source_classifier

    if model_name == 'wdgrl_michal':
        classifier = Network()
        classifier.load_state_dict(
            torch.load(f'{Config.wdgrl_michal_model_fp}_{particle_name}.pt',
                       map_location=torch.device(device)))
        return classifier

    else:
        source_classifier.load_state_dict(
            torch.load(f'{Config.source_model_fp}_{particle_name}.pt',
                       map_location=torch.device(device)))
        backbone = source_classifier.backbone

        particles_classifier = models_dict[model_name]['model']
        classifier = particles_classifier(backbone, Config.n_classes).to(device)
        classifier.load_state_dict(torch.load(f"{models_dict[model_name]['fp']}_{particle_name}.pt",
                                              map_location=torch.device(device)))

        return classifier