from tabulate import tabulate

from gradiend.evaluation.analyze_encoder import get_model_metrics
from gradiend.export import models as default_models


exported_metrics = {
    'acc_total': r'\accenc',
    'pearson_total': r'\corenc',
    #'pearson_total_p_value': r'\corenc p value',
    'acc': r'\accmf',
    'pearson_MF': r'\cormf',
    #'pearson_MF_p_value': r'\cormf p value',
    'encoded_abs_means_gender masked': r'\mamf',
    'encoded_abs_means_no gender masked': r'\masmf',
    'encoded_abs_means_no gender': '\man',
}

def print_encoder_stats(*models):
    if len(models) == 0:
        models = default_models
    elif len(models) == 1 and isinstance(models[0], dict):
        models = models[0]
    else:
        models = {model: model for model in models}

    csv_files = {pretty_model: rf'results/models/{model.removeprefix("results/models/")}_params_spl_test.csv' for model, pretty_model in models.items()}

    results = {}
    for model, file in csv_files.items():
        try:
            result = get_model_metrics(file)
            results[model] = result
        except FileNotFoundError:
            print(f'File {file} not found')

    # if a value contains of another dict, unwrap their values by concatenating the dict keys with '_'
    for file, metrics in results.items():
        for key, value in list(metrics.items()):
            if isinstance(value, dict):
                for k, v in value.items():
                    metrics[f'{key}_{k}'] = v
                del metrics[key]

    # Prepare data for tabulate
    headers = ['Model']
    for metric in exported_metrics.values():
        headers.append(metric)

    headers = [rf'\textbf{{{header}}}' for header in headers]


    table_data = []
    for model, metrics in sorted(results.items()):
        row = [model] + [f'{metrics.get(header, -1.0):.16f}' if 'p value' in header else metrics.get(header, '') for header in exported_metrics.keys()]
        table_data.append(row)

    # Print using tabulate
    print(tabulate(table_data, headers=headers, tablefmt='latex_raw', floatfmt=".3f"))

if __name__ == '__main__':
    print_encoder_stats(
        'bert-base-cased',
        'bert-large-cased',
        'distilbert-base-cased',
        'roberta-large',
        'gpt2',
        'Llama-3.2-3B',
        'Llama-3.2-3B-Instruct',
    )