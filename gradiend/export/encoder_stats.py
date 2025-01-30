from tabulate import tabulate

from gradiend.evaluation.analyze_encoder import get_model_metrics
from gradiend.export import models

exported_metrics = {
    'acc_total': r'\accenc',
    'pearson_total': r'\corenc',
    'acc': r'\accmf',
    'pearson_MF': r'\cormf',
    'encoded_abs_means_gender masked': r'\mamf',
    'encoded_abs_means_no gender masked': 'masmf',
    'encoded_abs_means_no gender': '\man',
}

csv_files = {pretty_model: rf'results/models/{model}_params_spl_test.csv' for model, pretty_model in models.items()}

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
    row = [model] + [metrics.get(header, '') for header in exported_metrics.keys()]
    table_data.append(row)

# Print using tabulate
print(tabulate(table_data, headers=headers, tablefmt='latex_raw', floatfmt=".3f"))

