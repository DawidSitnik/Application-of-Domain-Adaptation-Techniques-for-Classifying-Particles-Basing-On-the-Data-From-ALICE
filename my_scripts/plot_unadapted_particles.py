from utils.config import Config
import numpy as np
from utils.utils import load_pickle, save_pickle, does_file_exist
import umap
import matplotlib.pyplot as plt
import matplotlib.colors as col


def main():
    """
    - Plots an embedding of the output from feature extractor for all of the models
    and marks unadapted particles with red color and adapted with green.
    - Plots an embedding of the output from feature extractor for all of the models
    and marks domain with the color.

    Plots are saved into:
    - plots/umap/unadapted_particles/
    - plots/umap_embeddings/domain/

    Scripts which needs to be run before:
    1. All the training scripts from /training_models
    2. get_output_from_models_last_layer.py
    3. cluster_and_mark_unadapted_particles.py
    """
    print("Marking unadapted particles.")
    for particle_name in Config.particles_list:
        model_name = 'source'
        embedding_fp = f'{Config.source_fp}/pickles/umap_embeddings/{particle_name}_{model_name}_umap_embedding.pkl'
        if does_file_exist(embedding_fp):
            print(f'loading embedding for {particle_name}, {model_name}')
            embedding = load_pickle(embedding_fp)
        else:
            print(f'creating embedding for {particle_name}, {model_name}')
            # loading output from last layer
            test_source_features = load_pickle(
                f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_source_last_layer_output.pkl')
            test_target_features = load_pickle(
                f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_target_last_layer_output.pkl')
            # creating embedding
            features = np.concatenate((test_source_features, test_target_features), axis=0)
            embedding = umap.UMAP().fit_transform(features)

        test_source_features = load_pickle(
            f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_source_last_layer_output.pkl')
        test_target_features = load_pickle(
            f'{Config.source_fp}/pickles/last_layer_output/{model_name}_{particle_name}_target_last_layer_output.pkl')
        domains = np.concatenate((np.ones(len(test_source_features)), np.zeros(len(test_target_features))))

        # visualize and save umap with marked domains
        plt.figure(figsize=(30, 30))
        plt.scatter(embedding[:, 0], embedding[:, 1], c=domains, cmap=col.ListedColormap(["r", "b"]), s=2)
        plt.savefig(f'{Config.source_fp}/plots/umap_embeddings/domain/{particle_name}_{model_name}.png')
        save_pickle(embedding, embedding_fp)

        # load unadapted clusters list
        unadapted_clusters_list = load_pickle(f'{Config.source_fp}/pickles/unadapted_particles/{model_name}_{particle_name}_unadapted_clusters_list.pkl')
        # load features from feature extractor
        clustered_features = load_pickle(
            f'{Config.source_fp}/pickles/clustering_last_layer/{model_name}_{particle_name}_last_layer_clustering.pkl')

        # get indexes of unadapted particles
        unadapted_particles_indexes = clustered_features[clustered_features['cluster'].isin(unadapted_clusters_list)].index.values

        # plot embedding marking unadapted particles
        particles_labels = np.zeros(len(embedding))
        particles_labels[unadapted_particles_indexes] = 1
        plot_umap_with_unadapted_particles(embedding, particles_labels, particle_name, model_name)


def plot_umap_with_unadapted_particles(embedding, plot_labels, particles_name, model_name):
    color_map = col.ListedColormap(["turquoise", "r"])
    plt.figure(figsize=(30, 30))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=plot_labels, cmap=color_map, s=2)
    plt.title(f'Unadapted Particles ({particles_name}, {model_name})')
    plt.savefig(f'{Config.source_fp}/plots/umap_embeddings/unadapted_particles/{particles_name}_{model_name}.png')


if __name__ == '__main__':
    main()
