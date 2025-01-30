import json
import os

import inflect
import pandas as pd


def sanitize_split(split):
    if split == 'val':
        return 'validation'
    return split

def json_loads(x):
    if isinstance(x, (float, int)):
        return x

    try:
        return json.loads(x)
    except Exception:
        return [xx.removeprefix("'").removesuffix("'") for xx in x.removeprefix('[').removesuffix(']').split(',')]

def json_dumps(x):
    if isinstance(x, (float, int, str)):
        return x

    if isinstance(x, list) and len(x) == 1:
        return x[0]

    return json.dumps(x)

def get_default_prediction_file_name(model):
    model_name = model.name_or_path if hasattr(model, 'name_or_path') else model
    return f'data/cache/default_predictions_{model_name}.csv'

def get_available_models(prefix=''):
    # return all directories in results/models with the given prefix
    return [name for name in os.listdir('results/models') if os.path.isdir(f'results/models/{name}') and name.startswith(prefix)]


def read_default_predictions(model):
    file = get_default_prediction_file_name(model)
    try:
        cache_default_predictions = pd.read_csv(file)
        cache_default_predictions.set_index('text', inplace=True)
        cache_default_predictions['he'] = cache_default_predictions['he'].apply(json_loads)
        cache_default_predictions['she'] = cache_default_predictions['she'].apply(json_loads)
        cache_default_predictions['most_likely_token'] = cache_default_predictions['most_likely_token'].apply(json_loads)
        cache_default_predictions['label'] = cache_default_predictions['label'].apply(json_loads)
        cache_default_predictions_dict = cache_default_predictions.to_dict(orient='index')
    except FileNotFoundError:
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


def enrich_with_plurals(input_dict):
    # Create an inflect engine
    p = inflect.engine()

    # Iterate over the dictionary values
    for key, value in input_dict.items():
        # Enrich the list with plural forms
        plural_values = [p.plural(word) for word in value]
        # Update the dictionary with the enriched list
        input_dict[key] = list(set(plural_values + input_dict[key]))

    return input_dict
