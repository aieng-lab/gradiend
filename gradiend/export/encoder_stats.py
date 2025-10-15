import numpy as np
from tabulate import tabulate

from gradiend.evaluation.analyze_encoder import get_model_metrics
from gradiend.export import models as default_models
from gradiend.setups.race_religion.training import MuslimJewishSetup, ChristianMuslimSetup, ChristianJewishSetup, \
    WhiteAsianSetup, BlackAsianSetup, WhiteBlackSetup

exported_metrics = {
    'acc_total': r'\accenc',
    'pearson_total': r'\corenc',
    #'pearson_total_p_value': r'\corenc p value',
    'acc': r'\accmf',
    'pearson': r'\cormf',
    #'pearson_p_value': r'\cormf p value',
    'encoded_abs_means_gender masked': r'\mamf',
    'encoded_abs_means_no gender masked': r'\masmf',
    'encoded_abs_means_gerneutral': '\man',
}

def print_encoder_stats(*models):
    if len(models) == 0:
        models = default_models
    elif len(models) == 1 and isinstance(models[0], dict):
        models = models[0]
    else:
        models = {model: model for model in models}

    csv_files = {pretty_model: rf'results/models/gender-en/{model.removeprefix("results/models/")}_params_spl_test_v_3.csv' for model, pretty_model in models.items()}

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
        row = [model] + [f'{metrics.get(header, -1.0):.16f}' if 'p value' in header else metrics.get(header, f'missing entry {header}') for header in exported_metrics.keys()]
        table_data.append(row)

    # Print using tabulate
    print(tabulate(table_data, headers=headers, tablefmt='latex_raw', floatfmt=".3f"))

def print_encoder_stats_all_biases(*models):
    if len(models) == 0:
        models = default_models
    elif len(models) == 1 and isinstance(models[0], dict):
        models = models[0]
    else:
        models = {model: model for model in models}

    csv_files = {pretty_model: rf'results/models/{model.removeprefix("results/models/")}_params_spl_test_v_3.csv' for model, pretty_model in models.items()}

    results = {}
    for model, file in csv_files.items():
        try:
            result = get_model_metrics(file)
            results[model] = result
        except FileNotFoundError:
            print(f'File {file} not found')


    all_results = {
        'gender': results
    }

    setups = [
        BlackAsianSetup(),
        WhiteAsianSetup(),
        WhiteBlackSetup(),
        ChristianJewishSetup(),
        ChristianMuslimSetup(),
        MuslimJewishSetup(),
    ]

    for setup in setups:
        for model in models:
            model_id = f'results/models/{setup.id}/{model}-v5'
            try:
                result = setup.get_model_metrics(f'{model_id}/encoded_values.csv', plot=False)
                if setup.id not in all_results:
                    all_results[setup.id] = {}
                all_results[setup.id][model] = result
            except FileNotFoundError:
                print(f'File for setup {setup.id} and model {model} not found')

    headers = [
        #['', r'\textbf{Gender}', r'\multicolumn{3}{c}{\textbf{Race}', r'\multicolumn{3}{c}{\textbf{Religion}}'],
        ['', 'MF', 'AB', 'AC', 'BC', 'CJ', 'CM', 'MJ'],
    ]

    def get_score(*keys):
        score = all_results
        for key in keys:
            if key in score:
                score = score[key]
            else:
                return -1.0
        return score * 100 if isinstance(score, float) else -1.0

    table_data = []
    for model in models:
        row = [default_models.get(model, model)]

        # add metric for gender
        score = get_score('gender', model, 'pearson')
        row.append(get_score('gender', model, 'pearson'))
        row.append(get_score('gender', model, 'pearson_total'))

        #row.append(get_score('race_white_black', 'corr_white<->black'))

        row.append(get_score('race_black_asian', model, 'corr_asian<->black'))
        row.append(get_score('race_black_asian', model, 'corr'))
        row.append(get_score('race_white_asian', model, 'corr_asian<->white'))
        row.append(get_score('race_white_asian', model, 'corr'))
        row.append(get_score('race_white_black', model, 'corr_black<->white'))
        row.append(get_score('race_white_black', model, 'corr'))
        row.append(get_score('religion_christian_jewish', model, 'corr_christian<->jewish'))
        row.append(get_score('religion_christian_jewish', model, 'corr'))
        row.append(get_score('religion_christian_muslim', model, 'corr_christian<->muslim'))
        row.append(get_score('religion_christian_muslim', model, 'corr'))
        row.append(get_score('religion_muslim_jewish', model, 'corr_jewish<->muslim'))
        row.append(get_score('religion_muslim_jewish', model, 'corr'))


        table_data.append(row)

    # convert to numpy array for processing
    arr = np.array(table_data, dtype=object)

    # --- Bold best values per metric column ---
    fmt = '{:.1f}'
    def fmtat(x):
        return fmt.format(x)
    for col in range(1, arr.shape[1]):
        col_values = arr[:, col].astype(float)
        best_val = np.nanmax(col_values)
        for i, val in enumerate(col_values):
            if fmtat(val) == fmtat(best_val) and val > 0:
                arr[i, col] = "\\textbf{" + f"{val:.1f}" + "}"
            else:
                arr[i, col] = f"{val:.1f}"

    # --- Add two mean columns: mean of odd cols (A) and even cols (B) ---
    mean_A = []
    mean_B = []
    for row in table_data:
        vals = np.array(row[1:], dtype=float)
        A_vals = vals[0::2]  # odd indices
        B_vals = vals[1::2]  # even indices
        # exclude -1
        A_vals = A_vals[A_vals > 0]
        B_vals = B_vals[B_vals > 0]
        mean_A.append(np.nanmean(A_vals))
        mean_B.append(np.nanmean(B_vals))

    arr_with_means = []
    for i, row in enumerate(arr.tolist()):
        row.append(f"{mean_A[i]:.1f}")
        row.append(f"{mean_B[i]:.1f}")
        arr_with_means.append(row)

    # --- Bold the best mean_A and mean_B across models ---
    best_A = np.nanmax(mean_A)
    best_B = np.nanmax(mean_B)
    for i in range(len(arr_with_means)):
        if fmtat(mean_A[i]) == fmtat(best_A):
            arr_with_means[i][-2] = "\\textbf{" + arr_with_means[i][-2] + "}"
        if fmtat(mean_B[i]) == fmtat(best_B):
            arr_with_means[i][-1] = "\\textbf{" + arr_with_means[i][-1] + "}"

    # --- Add mean row (across models) ---
    numeric_arr = np.array(table_data, dtype=object)
    mean_row_A = np.nanmean([np.nanmean(r[1:][0::2].astype(float)) for r in numeric_arr])
    mean_row_B = np.nanmean([np.nanmean(r[1:][1::2].astype(float)) for r in numeric_arr])
    mean_row_vals = [np.nanmean(numeric_arr[:, col].astype(float)) for col in range(1, numeric_arr.shape[1])]
    mean_row = ["Mean"] + [f"{v:.1f}" for v in mean_row_vals] + [f"{mean_row_A:.1f}", f"{mean_row_B:.1f}"]

    arr_with_means.append(mean_row)

    # --- Headers ---
    headers = ["Model"] + [f"Col{i}" for i in range(1, arr.shape[1])] + ["Mean-A", "Mean-B"]

    print(tabulate(arr_with_means, headers=headers, tablefmt='latex_raw', stralign="center"))



if __name__ == '__main__':
    print_encoder_stats_all_biases(
        'bert-base-cased',
        'bert-large-cased',
        'distilbert-base-cased',
        'roberta-large',
        'gpt2',
        'Llama-3.2-3B',
        'Llama-3.2-3B-Instruct',
    )