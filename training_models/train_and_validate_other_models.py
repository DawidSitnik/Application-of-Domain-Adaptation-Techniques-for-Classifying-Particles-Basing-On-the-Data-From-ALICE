from xgboost import XGBClassifier
from utils.config import Config
from utils.utils import load_pickle, save_pickle, does_file_exist
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier


def main():

    models_dict = {
        'xgb': XGBClassifier(),
        'catboost': CatBoostClassifier(),
        'random_forest': RandomForestClassifier()
    }

    for particle_code in Config.particles_dict:
        for model_name in models_dict:
            model = models_dict[model_name]
            particle_name = Config.particles_dict[particle_code]
            datasets_dict = load_pickle(
                f'{Config.training_data_fp}/datasets_dict_{particle_name}.pkl')
            fp = get_fp(model_name, particle_name)

            if does_file_exist(fp):
                print(f'Loading {model_name} model for {particle_name}')
                model = load_pickle(fp)
            else:
                print(f'Creating {model_name} model for {particle_name}')
                model.fit(datasets_dict['x_train_source'], datasets_dict['y_train_source'])
                save_pickle(model, fp)

            auc_score = validate_xgb(model, datasets_dict['x_test_target'], datasets_dict['y_test_target'])
            print(f'AUC for {model_name}, {particle_name}: {auc_score}')
        print('\n')


def validate_xgb(model, x_test, y_test):
    predictions = model.predict(x_test)
    return average_precision_score(y_test, predictions)


def get_fp(model_name, particle_name):
    if model_name == 'catboost':
        return f'{Config.cb_model_fp}_{particle_name}.pt'
    if model_name == 'xgb':
        return f'{Config.xgb_model_fp}_{particle_name}.pt'
    if model_name == 'random_forest':
        return f'{Config.random_forest_model_fp}_{particle_name}.pt'


if __name__ == '__main__':
    main()
