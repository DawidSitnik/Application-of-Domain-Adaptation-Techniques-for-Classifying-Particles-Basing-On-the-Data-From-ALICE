from utils.config import Config
from utils.utils import load_pickle, save_pickle, get_classifier
from utils.clustering import cluster_df
import numpy as np
import pandas as pd
from utils.validation import validate
from sklearn.metrics import average_precision_score
from utils.models_dict import models_dict
from utils.fid import calculate_fid
from utils.utils import does_file_exist, get_dataset
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

MAX_DIFFUSION = 100
SOM_SIZE = 20
SOM_N_ITER = 3000
SOM_LEARNING_RATE = 0.001


def main():
    """
    Makes clustering on the output from feature extractor of each model
    and prepare a summary of each cluster.
    If the pickle with clustering already exists, clusters are read from a pickle.
    If not, a new clustering is computed.

    Steps made in that script:
        - loading data
        - clustering - assigning cluster to each particle and saves particles with
                       assigned clusters to pickles/clustering_last_layer/
        - creating clusters summaries and saving it into tables/cluster_summaries/
        - creating list of unadapted clusters and saves it into /pickles/unadapted_particles
        - saving unadapted particles with features from feature extractor into /pickles/unadapted_particles


    Summary consists of:
        - count - amount of particles in a cluster (target+source)
        - count_source - amount of source particles in a cluster
        - count_target - amount of target particles in a cluster
        - diffusion - max(source particles, target particles) / min(source particles, target particles) in a cluster
        - precision_recall - precision recall score for each cluster (target+source)
        - precision_recall_source - precision recall score for source particles in each cluster
        - precision_recall_target - precision recall score for target particles in each cluster

    Scripts which needs to be run before:
    1. All the training scripts from /training_models
    2. get_output_from_models_last_layer.py
    """
    # Config.particles_list = ['protons']
    model_name = 'source'
    for i, particle_name in enumerate(Config.particles_list):
        datasets_dict = load_pickle(f'{Config.training_data_fp}/datasets_dict_{particle_name}.pkl')

        # load data
        print(f'\n{particle_name}, {model_name}')
        test_source_fp = f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_source_last_layer_output.pkl'
        test_source_features = load_pickle(test_source_fp)

        # clustering
        clustered_features_fp = f'{Config.source_fp}/pickles/clustering_last_layer/{model_name}_{particle_name}_last_layer_clustering.pkl'
        if does_file_exist(clustered_features_fp):
            print('loading clusters')
            clustered_features = load_pickle(clustered_features_fp)
        else:
            print('creating new clusters')
            test_target_features = load_pickle(f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_target_last_layer_output.pkl')
            features = np.concatenate((test_source_features, test_target_features), axis=0)
            clustered_features = cluster_df(som_width=SOM_SIZE, som_height=SOM_SIZE, df=features, n_iter=SOM_N_ITER, learning_rate=SOM_LEARNING_RATE)
            save_pickle(clustered_features, clustered_features_fp)

        print('making predictions')
        model = get_classifier(model_name, particle_name, models_dict)
        _, outputs_source, targets_source = validate(datasets_dict['test_source_loader'], model, return_predictions=True, return_probabilities=True)
        _, outputs_target, targets_target = validate(datasets_dict['test_target_loader'], model, return_predictions=True, return_probabilities=True)
        clustered_features['output'] = list(outputs_source) + list(outputs_target)
        clustered_features['target'] = list(targets_source) + list(targets_target)

        # create or load model for classifying domain basing on model's latent space
        domain_classifier_fp = f'{Config.source_fp}/pickles/domain_classifiers/{model_name}_{particle_name}_domain_classifier.pkl'
        if does_file_exist(domain_classifier_fp):
            domain_classifier = load_pickle(domain_classifier_fp)
            print(f'\nloading domain classifier for: {particle_name}, {model_name}')
        else:
            print(f'\ncreating domain classifier for: {particle_name}, {model_name}')
            test_source_features = load_pickle(
                f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_source_last_layer_output.pkl')
            test_target_features = load_pickle(
                f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_target_last_layer_output.pkl')

            dataset = get_dataset(test_source_features, test_target_features)
            domain_classifier = get_domain_classifier(dataset)
            save_pickle(domain_classifier, domain_classifier_fp)

        # creating clusters summary
        clusters_summary_fp = f'{Config.source_fp}/tables/cluster_summaries/{model_name}_{particle_name}_clusters_summaries.csv'
        if does_file_exist(clusters_summary_fp):
            print('reading summary')
            cluster_summary = pd.read_csv(clusters_summary_fp)
        else:
            print('creating summary')
            cluster_summary = pd.DataFrame({'count': clustered_features.groupby('cluster').size(),
                                            'count_source': clustered_features[:len(test_source_features)].groupby('cluster').size(),
                                            'count_target': clustered_features[len(test_source_features):].groupby('cluster').size(),
                                            'cluster': clustered_features.groupby('cluster')['cluster'].mean(),
                                            'fid': clustered_features.drop(columns=['target', 'output']).groupby('cluster').apply(
                lambda cluster: calculate_fid_lambda(cluster, len(test_source_features))),
                                            'a-distance': clustered_features.drop(columns=['target', 'output']).groupby('cluster').apply(
                lambda cluster: get_cluster_a_distance(cluster, domain_classifier, len(test_source_features))),
                                            'diffusion': clustered_features.groupby('cluster').apply(
                lambda cluster: get_cluster_diffusion(cluster, len(test_source_features)))
                #                             'precision_recall': clustered_features.groupby('cluster').apply(
                # lambda cluster: get_cluster_precision_recall(cluster)),
                #                             'precision_recall_source': clustered_features[:len(test_source_features)].groupby('cluster').apply(
                # lambda cluster: get_cluster_precision_recall(cluster)),
                #                             'precision_recall_target': clustered_features[len(test_source_features):].groupby('cluster').apply(
                # lambda cluster: get_cluster_precision_recall(cluster))
                }).reset_index(drop=True)

            print('saving to csv')
            cluster_summary.to_csv(clusters_summary_fp)

        # save unadapted clusters list
        index_of_biggest_cluster = cluster_summary['count'].argmax()
        metric = Config.metric
        diffusion_of_biggest_cluster = cluster_summary.iloc[index_of_biggest_cluster][metric]

        unadapted_clusters_list = list(set(cluster_summary[cluster_summary[metric] > diffusion_of_biggest_cluster]['cluster']))
        save_pickle(unadapted_clusters_list,
                    f'{Config.source_fp}/pickles/unadapted_particles/{model_name}_{particle_name}_unadapted_clusters_list_{metric}.pkl')

        # save unadapted_particles
        unadapted_particles_indexes = clustered_features[
            clustered_features['cluster'].isin(unadapted_clusters_list)].index.values
        unadapted_particles_source = datasets_dict['x_test_source'].loc[
            datasets_dict['x_test_source'].index.isin(unadapted_particles_indexes)]
        unadapted_particles_target = datasets_dict['x_test_target'].loc[
            datasets_dict['x_test_target'].index.isin(unadapted_particles_indexes)]

        save_pickle(unadapted_particles_source,
                    f'{Config.source_fp}/pickles/unadapted_particles/{particle_name}_{model_name}_unadapted_particles_source.pkl')
        save_pickle(unadapted_particles_target,
                    f'{Config.source_fp}/pickles/unadapted_particles/{particle_name}_{model_name}_unadapted_particles_target.pkl')

        # print summary
        n_unadapted_particles = \
            clustered_features[clustered_features['cluster'].isin(unadapted_clusters_list)].groupby('cluster').count()[
                0].sort_values().sum()
        print(f'Percentage of unadapted particles: {n_unadapted_particles/len(clustered_features)*100}')


def get_domain_classifier(dataset):
    classifier = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    x_train, x_test, y_train, y_test = train_test_split(dataset.drop(columns=['domain']), dataset['domain'],
                                                        test_size=0.3, random_state=0)
    classifier.fit(x_train, y_train)

    return classifier


def get_cluster_a_distance(cluster, domain_classifier, source_df_len):
    cluster = cluster.drop(columns=['cluster'])
    y_pred = domain_classifier.predict(cluster)
    cluster['new_index'] = cluster.index
    cluster['domain'] = cluster.new_index.apply(lambda x: 0 if x < source_df_len else 1)
    my_distance = 1 - average_precision_score(cluster['domain'], y_pred)
    return my_distance



def get_cluster_diffusion(cluster, source_df_len):
    """
    returns ratio of particles from source and target domains
    """
    indexes_list = list(cluster.index.values)
    source_indexes_len = len([x for x in indexes_list if x < source_df_len])
    target_indexes_len = len([x for x in indexes_list if x >= source_df_len])

    if min(source_indexes_len, target_indexes_len) == 0:
        if max(source_indexes_len, target_indexes_len) == 0:
            ratio = 1
        else:
            ratio = MAX_DIFFUSION
    else:
        ratio = max(source_indexes_len, target_indexes_len) / min(source_indexes_len, target_indexes_len)

    return ratio


def split_cluster_into_source_and_target(cluster, source_df_len):
    source_df = cluster[cluster.index < source_df_len]
    target_df = cluster[cluster.index >= source_df_len]
    return source_df, target_df


def get_cluster_precision_recall(cluster):
    target = list(cluster['target'])
    output = list(cluster['output'])
    try:
        return average_precision_score(target, output)
    except:
        return None


def calculate_fid_lambda(cluster, source_len):
    source = cluster.iloc[cluster.index < source_len]
    target = cluster.iloc[cluster.index >= source_len]
    if min(len(source), len(target)) < 40:
        return None
    else:
        return calculate_fid(source, target)


if __name__ == '__main__':
    main()
