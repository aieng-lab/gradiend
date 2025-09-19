import os

import pandas as pd

from gradiend.data.util import json_dumps



def get_available_models(prefix=''):
    # return all directories in results/models with the given prefix
    return [name for name in os.listdir('results/models') if os.path.isdir(f'results/models/{name}') and name.startswith(prefix)]

def get_default_prediction_file_name(model):
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else model
    return f'results/cache/default_predictions_v2_{model_name}.csv'

def read_default_predictions(model):
    from pympler.util.bottle import json_loads
    file = get_default_prediction_file_name(model)
    try:
        cache_default_predictions = pd.read_csv(file)
        cache_default_predictions.set_index('text', inplace=True)
        cache_default_predictions['he'] = cache_default_predictions['he'].apply(json_loads)
        cache_default_predictions['she'] = cache_default_predictions['she'].apply(json_loads)
        cache_default_predictions['most_likely_token'] = cache_default_predictions['most_likely_token'].apply(json_loads)
        cache_default_predictions['label'] = cache_default_predictions['label'].apply(json_loads)
        cache_default_predictions_dict = cache_default_predictions.to_dict(orient='index')
    except Exception:
        cache_default_predictions_dict = {}
    return cache_default_predictions_dict


def write_default_predictions(default_predictions, model):
    file = get_default_prediction_file_name(model)
    # Ensure the directory exists
    directory = os.path.dirname(file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    cache_default_predictions = pd.DataFrame.from_dict(default_predictions, orient='index')
    cache_default_predictions['he'] = cache_default_predictions['he'].apply(json_dumps)
    cache_default_predictions['she'] = cache_default_predictions['she'].apply(json_dumps)
    cache_default_predictions['most_likely_token'] = cache_default_predictions['most_likely_token'].apply(json_dumps)
    cache_default_predictions['label'] = cache_default_predictions['label'].apply(json_dumps)
    cache_default_predictions.reset_index(inplace=True)
    cache_default_predictions.rename(columns={'index': 'text'}, inplace=True)
    cache_default_predictions.to_csv(file, index=False)


gender_pronouns = ['him', 'her', 'his', 'hers', 'himself', 'herself', 'he', 'she']

