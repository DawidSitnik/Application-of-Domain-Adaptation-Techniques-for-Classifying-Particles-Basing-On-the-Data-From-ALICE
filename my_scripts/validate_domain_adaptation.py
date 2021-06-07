from utils.config import Config
from utils.utils import load_pickle, save_pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from utils.models_dict import models_dict
from utils.utils import does_file_exist, get_dataset


def main():
    """
    Calculate domain adaptation quality for each model and saves the outputs
    into /pickles/domain_adaptation_quality.

    To measure adaptation quality the metric which equals to (1 - A) was used.
    In that case A is the AUC of XGB classifier trained on the binary problem
    of discriminating the source and target. Greater A means easier distinction
    between two domains, which is indication of low adaptation quality. It was
    decided to use 1 - A, as a greater value of that metric means better
    adaptation and it is more intuitive.

    Scripts which needs to be run before:
    1. All the training scripts from /training_models
    2. get_output_from_models_last_layer.py
    """
    results_dict = {'wdgrl': {}, 'source': {}, 'cdan': {}, 'dan': {}, 'dann': {}, 'jan': {}, 'mdd': {}}

    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    for i, particle_name in enumerate(Config.particles_list):
        for model_name in models_dict:
            auc_pickle_fp = f'{Config.source_fp}/pickles/domain_adaptation_quality/{model_name}_{particle_name}_auc.pkl'
            if does_file_exist(auc_pickle_fp):
                precision_recal_auc = load_pickle(auc_pickle_fp)
                print(f'\nloading model for: {particle_name}, {model_name}')
            else:
                print(f'\ncreating model for: {particle_name}, {model_name}')
                test_source_features = load_pickle(
                f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_source_last_layer_output.pkl')
                test_target_features = load_pickle(
                    f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_target_last_layer_output.pkl')

                dataset = get_dataset(test_source_features, test_target_features)
                precision_recal_auc = get_auc(xgb, dataset)

                save_pickle(precision_recal_auc, auc_pickle_fp)

            results_dict[model_name][particle_name] = 1 - precision_recal_auc
            print(f'{model_name}, {particle_name}: {precision_recal_auc}')

    results_df = pd.DataFrame([[model_name] + list(results_dict[model_name].values()) for model_name in results_dict],
                              columns=['Model Name', 'Electrons', 'Pions', 'Kaons', 'Protons'])
    results_df['mean'] = results_df[['Electrons', 'Kaons', 'Pions', 'Protons']].apply(
        lambda x: (x[0] + x[1] + x[2] + x[3]) / 4, axis=1)
    results_df = results_df.sort_values(by='mean', ascending=True)
    print(results_df)
    save_pickle(results_df, f'{Config.source_fp}/pickles/domain_adaptation_quality/domain_adaptation_quality.pkl')


def get_auc(classifier, dataset):
    x_train, x_test, y_train, y_test = train_test_split(dataset.drop(columns=['domain']), dataset['domain'],
                                                        test_size=0.3, random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    return average_precision_score(y_test, y_pred)


if __name__ == '__main__':
    main()
