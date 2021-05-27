from utils.config import Config
from utils.utils import load_pickle
import pandas as pd
import matplotlib.pyplot as plt


def main():
    """
    Compares classification and adaptation quality for all models.
    - reads adaptation quality from pickles/domain_adaptation_quality/domain_adaptation_quality.pkl
    - reads classification quality from /pickles/classification_quality/classification_quality_df.pkl
    - saves comparisons into plots/classification_vs_adaptation_quality

    Scripts which needs to be run before:
    1. All the training scripts from /training_models
    2. get_output_from_models_last_layer.py
    3. validate_domain_adaptation.py
    """
    model_suffix = '_model'
    da_suffix = '_da'

    adaptation_quality_df = load_pickle(
        f'{Config.source_fp}/pickles/domain_adaptation_quality/domain_adaptation_quality.pkl')
    classification_quality_df = load_pickle(
        f'{Config.source_fp}/pickles/classification_quality/classification_quality_df.pkl')

    print("model quality")
    print(classification_quality_df.sort_values(by='mean', ascending=True))
    print("domain adaptation quality")
    print(adaptation_quality_df)

    df_merged = pd.merge(adaptation_quality_df, classification_quality_df, left_on='Model Name', right_on='Model Name', how='left', suffixes=[model_suffix, da_suffix])
    print(df_merged)

    for column in classification_quality_df.columns:
        if column != "Model Name":
            column_1 = column + model_suffix
            column_2 = column + da_suffix

            df_to_print = df_merged[[column_1, column_2, 'Model Name']].sort_values(by=column_1, ascending=True)
            plt.plot(df_to_print[column_1], df_to_print[column_2])
            plt.scatter(df_to_print[column_1], df_to_print[column_2])
            plt.ylabel('AUC for particle classifier')
            plt.xlabel('1 - AUC for domain classifier')
            plt.title(f'Classification vs Adaptation Quality for {column}')

            # appending annotations
            for i, model_name in enumerate(list(df_to_print['Model Name'])):
                plt.annotate(model_name, (list(df_to_print[column_1])[i], list(df_to_print[column_2])[i]))
            plt.savefig(f'{Config.source_fp}/plots/classification_vs_adaptation_quality/{column}.png')


if __name__ == '__main__':
    main()
