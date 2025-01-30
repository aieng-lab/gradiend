import json
import os
import time
from collections import defaultdict

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from torch import softmax
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from gradiend.data import read_namextend, read_namexact, read_gentypes, read_geneutral
from mlm import evaluate_mlm
from gradiend.model import ModelWithGradiend
from gradiend.util import hash_model_weights


def calculate_average_probability_difference(fairness_dict):
    total_diff = defaultdict(float)
    num_texts = len(fairness_dict)

    for text, probs in fairness_dict.items():
        prob_m = probs['M']
        prob_f = probs['F']
        for key in prob_m:
            total_diff[key] += abs(prob_m[key] - prob_f[key])

    for key in total_diff:
        total_diff[key] /= num_texts
    return total_diff


def calculate_average_prediction_quality(gender_probabilities):
    num_texts = len(gender_probabilities)
    keys = list(gender_probabilities.values())[0]['M'].keys()
    total_sums = {key: sum([sum([v[key] for v in probs.values() if isinstance(v, dict)]) for probs in gender_probabilities.values()]) for key in keys}
    averages = {key: total_sums[key] / num_texts for key in keys}
    return averages


def calculate_baseline_change(current_model, baseline_model, type='total'):
    gender_preference_accuracy = 0
    overall_accuracy = 0
    absolute_difference_sum = 0

    total_texts = len(current_model)

    for text in current_model:
        current_M = current_model[text]['M'][type]
        current_F = current_model[text]['F'][type]
        baseline_M = baseline_model[text]['M'][type]
        baseline_F = baseline_model[text]['F'][type]

        # Gender Preference Accuracy
        current_prefers_M = current_M > current_F
        baseline_prefers_M = baseline_M > baseline_F
        if current_prefers_M == baseline_prefers_M:
            gender_preference_accuracy += 1

        # Overall Accuracy
        current_max_gender = 'M' if current_M > current_F else 'F'
        baseline_max_gender = 'M' if baseline_M > baseline_F else 'F'
        if current_max_gender == baseline_max_gender:
            overall_accuracy += 1

        # Absolute Difference in Preference
        current_diff = current_M - current_F
        baseline_diff = baseline_M - baseline_F
        absolute_difference_sum += abs(current_diff - baseline_diff)

    metrics = {
        "gender_preference_accuracy": gender_preference_accuracy / total_texts,
        "overall_accuracy": overall_accuracy / total_texts,
        "average_absolute_difference": absolute_difference_sum / total_texts
    }

    return metrics


def compute_gender_preference_accuracy(current_model: dict) -> float:
    """
    Computes the accuracy of how often the current model prefers the male gender over the female gender.
    """
    male_preference_count = 0
    total_texts = len(current_model)

    for text in current_model:
        current_M = current_model[text]['M']['total']
        current_F = current_model[text]['F']['total']

        if current_M > current_F:
            male_preference_count += 1

    return male_preference_count / total_texts


def evaluate_gender_bias_name_predictions(model, tokenizer, text_prefix='', batch_size=64, df_name=None, baseline=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    data = read_gentypes()
    names_df = read_namextend()

    names_df['name_lower'] = names_df['name'].str.lower()
    names_to_split = names_df.set_index('name_lower')['split'].to_dict()
    gender_mapping_he = names_df[names_df['gender'] == 'M'].set_index('name_lower')['prob_M'].to_dict()
    gender_mapping_she = names_df[names_df['gender'] == 'F'].set_index('name_lower')['prob_F'].to_dict()

    tokenizer_vocab_lower = [k.lower() for k in tokenizer.vocab.keys()]
    gender_mapping_he = {k: v for k, v in gender_mapping_he.items() if k.lower() in tokenizer_vocab_lower}
    gender_mapping_she = {k: v for k, v in gender_mapping_she.items() if k.lower() in tokenizer_vocab_lower}

    if not text_prefix:
        gender_mapping_he['he'] = 1.0
        gender_mapping_she['she'] = 1.0

        if 'He' in tokenizer.vocab:
            gender_mapping_he['He'] = 1.0
        if 'She' in tokenizer.vocab:
            gender_mapping_she['She'] = 1.0

    #he_token_indices = [tokenizer.vocab[name] for name in gender_mapping_he]
    he_tokens = [name for name in tokenizer.vocab if name.lower() in gender_mapping_he]
    he_token_indices = [tokenizer.vocab[name] for name in he_tokens]
    he_token_factors = np.array([gender_mapping_he[name.lower()] for name in he_tokens])

    #she_token_indices = [tokenizer.vocab[name] for name in gender_mapping_she]
    she_tokens = [name for name in tokenizer.vocab if name.lower() in gender_mapping_she]
    she_token_indices = [tokenizer.vocab[name] for name in she_tokens]
    she_token_factors = np.array([gender_mapping_she[name.lower()] for name in she_tokens])

    # todo this is not really a good check to include he/she as gender specific words but works effectively

    gender_probabilities = {}
    gender_data = {key: [] for key in
                   ['text', 'gender', 'token', 'probability', 'prob_he', 'prob_she', 'most_likely_token', 'split']}

    all_texts = []
    for _, record in data.iterrows():
        text = record['text']
        if text_prefix:
            if text.startswith('[NAME]'):
                text = f'{text_prefix.strip()}, [NAME], {text.removeprefix("[NAME]").strip()}'
            else:
                text = f'{text_prefix.strip()} {text}'

        masked_text = text.replace("[NAME]", tokenizer.mask_token)
        all_texts.append(masked_text)

    vocab = {v: k for k, v in tokenizer.vocab.items()}
    token_idx_he = tokenizer.vocab['he'] # todo He, She
    token_idx_she = tokenizer.vocab['she']

    for start_idx in range(0, len(all_texts), batch_size):
        end_idx = min(start_idx + batch_size, len(all_texts))
        batch_texts = all_texts[start_idx:end_idx]

        # Tokenize the batch
        batch_tokenized_text = tokenizer(batch_texts, padding=True, return_tensors="pt", truncation=True)
        input_ids = batch_tokenized_text["input_ids"].to(device)
        attention_mask = batch_tokenized_text["attention_mask"].to(device)

        # Find mask token index in each input
        mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

        for i in range(len(batch_texts)):
            text = batch_texts[i]
            text_dict = {key: {} for key in {'M', 'F'}}
            masked_index = mask_token_index[1][i].item()
            predictions = logits[i, masked_index]

            softmax_probs = softmax(predictions, dim=-1).cpu()

            most_likely_token_id = torch.argmax(softmax_probs).item()
            most_likely_token = tokenizer.decode([most_likely_token_id])
            prob_he = softmax_probs[token_idx_he].item()
            prob_she = softmax_probs[token_idx_she].item()

            for gender, token_indices, token_factors in [('M', he_token_indices, he_token_factors), ('F', she_token_indices, she_token_factors)]:

                gender_probs = softmax_probs[token_indices] * token_factors

                relevant_token_indices = token_indices # [i for i, prob in zip(token_indices, gender_probs) if prob > 0.0001]
                tokens = [vocab[i] for i in relevant_token_indices]
                splits = np.array([names_to_split.get(token) for token in tokens])

                n = len(tokens)
                gender_data['text'] += n * [text]
                gender_data['gender'] += n * [gender]
                gender_data['token'] += tokens
                gender_data['probability'] += gender_probs.tolist()
                gender_data['prob_he'] += n * [prob_he]
                gender_data['prob_she'] += n * [prob_she]
                gender_data['most_likely_token'] += n * [most_likely_token]
                gender_data['split'] += splits.tolist()
                gender_prob = gender_probs.sum().item()

                text_dict[gender] = {}
                text_dict[gender]['total'] = gender_prob
                for split in ['test', 'train', 'val']:
                    ids = splits == split
                    gender_prob_split = gender_probs[ids].sum().item()
                    text_dict[gender][split] = gender_prob_split

            keys = text_dict[gender].keys()
            sums = {key: sum(text_dict[g][key] for g in text_dict) for key in keys}

            factor_M = {key: text_dict['M'][key] / sums[key] if sums[key] > 0 else 0 for key in keys}
            factor_F = {key: text_dict['F'][key] / sums[key] if sums[key] > 0 else 0 for key in keys}
            factor_max = {key: max(factor_M[key], factor_F[key]) for key in keys}
            text_dict['factor_M'] = factor_M
            text_dict['factor_F'] = factor_F
            text_dict['factor_max'] = factor_max
            sum_M = text_dict['M']['total']
            sum_F = text_dict['F']['total']
            text_apd = min(1.0, max(0.0, abs(sum_M - sum_F)))
            text_apd = abs(sum_M - sum_F)
            if not (0 <= text_apd <= 1):
                raise ValueError(f"Invalid APD: {text_apd}")

            text_dict['text_apd'] = text_apd
            text_dict['text_bpi'] = (1 - text_apd) * (sum_M + sum_F)
            text_dict['text_mpi'] = (1 - sum_F) * sum_M
            text_dict['text_fpi'] = (1 - sum_M) * sum_F

            if not (0 <= text_dict['text_bpi'] <= 1):
                raise ValueError(f"Invalid BPI: {text_dict['text_bpi']}")

            gender_probabilities[text] = text_dict


    apd = calculate_average_probability_difference(gender_probabilities)
    _bpi = np.mean([prob['text_bpi'] for prob in gender_probabilities.values()]).item()
    _mpi = np.mean([prob['text_mpi'] for prob in gender_probabilities.values()]).item()
    _fpi = np.mean([prob['text_fpi'] for prob in gender_probabilities.values()]).item()

    prediction_quality = calculate_average_prediction_quality(gender_probabilities)
    if isinstance(baseline, dict):
        baseline_change = calculate_baseline_change(gender_probabilities, baseline)
    else:
        baseline_change = None

    df = pd.DataFrame.from_dict(gender_data)
    if df_name:
        file_name = f'results/models/metrics/{df_name.removeprefix("results/models/").removesuffix(".csv")}.csv'
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        df.to_csv(file_name, index=False)

    # calculate std of probabilities per gender as measurement how unbiased names are
    stds = {}
    for gender, gender_df in df.groupby('gender'):
        prob_key = 'prob_he' if gender == 'M' else 'prob_she'
        std = gender_df[prob_key].std()
        stds[f'std_{gender}'] = std
    std = np.mean(list(stds.values())).item()

    # Calculate average probabilities for each gender
    keys = list(gender_probabilities.values())[0]['M'].keys()
    avg_prob_m = {key: sum(probs['M'][key] for probs in gender_probabilities.values()) / len(gender_probabilities) for key in keys}
    avg_prob_f = {key: sum(probs['F'][key] for probs in gender_probabilities.values()) / len(gender_probabilities) for key in keys}
    he_prob = compute_gender_preference_accuracy(gender_probabilities)

    # Calculate preference score
    preference_score = {key: abs(avg_prob_m[key] - avg_prob_f[key]) for key in keys}

    result = {
        'apd': apd,
        'pq': prediction_quality,
        '_bpi': _bpi,
        '_mpi': _mpi,
        '_fpi': _fpi,
        'avg_prob_m': avg_prob_m,
        'avg_prob_f': avg_prob_f,
        'preference_score': preference_score,
        'he_prob': he_prob,
        'std': std,
        **stds,
    }

    if baseline is True:
        result['baseline_change'] = baseline_change
        return result, gender_probabilities

    return result


def evaluate_non_gender_mlm(model, tokenizer, max_size=1000):
    df = read_geneutral(split='test', max_size=max_size)
    texts = df['text'].tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    result, stats = evaluate_mlm(model, tokenizer, texts, verbose=False)
    return result



# we evaluate the model on different tasks
# - MLM on non gender data
# - MLM on gender data with pronouns masked (assuming the model changes preference to one gender, evaluate the success of this approach)
# - MLM on
# - evaluate on special data? Sentences with typical gender bias, with the names masked, and evaluate what genders the names have the model predicts
#       -
def evaluate_model(model, tokenizer, verbose=True, df_name=None, thorough=True, force=False, only_friend=True):
    model_hash = hash_model_weights(model)
    cache_file = f'results/cache/evaluate_model/{model_hash}.json'
    if not force:
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            pass

    # Measure the start time of the entire evaluation
    total_start_time = time.time()
    result = {}

    # Evaluate non-gender MLM stats and measure time
    start_time = time.time()
    max_size = 1000 if thorough else 25 # todo set to 10000?
    non_gender_mlm_stats = evaluate_non_gender_mlm(model, tokenizer, max_size=max_size)
    non_gender_mlm_time = time.time() - start_time
    result['mlm'] = non_gender_mlm_stats

    if not only_friend:
        # Evaluate gender bias stats and measure time
        start_time = time.time()
        gender_bias_stats = evaluate_gender_bias_name_predictions(model, tokenizer, df_name=df_name)
        gender_bias_time = time.time() - start_time
        result['gender_bias'] = gender_bias_stats

    # Evaluate gender bias name stats with prefix and measure time
    start_time = time.time()
    if df_name:
        df_name = f'{df_name.replace(".csv", "")}_friend'
    gender_bias_name_stats = evaluate_gender_bias_name_predictions(model, tokenizer, text_prefix='My friend', df_name=df_name)
    gender_bias_name_time = time.time() - start_time
    result['gender_bias_names'] = gender_bias_name_stats

    # Measure the end time of the entire evaluation
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Calculate relative times
    non_gender_mlm_relative = non_gender_mlm_time / total_time
    gender_bias_name_relative = gender_bias_name_time / total_time

    if verbose:
        # Print nicely formatted times
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(
            f"Non-gender MLM evaluation time: {non_gender_mlm_time:.2f} seconds ({non_gender_mlm_relative:.2%} of total time)")
        if not only_friend:
            gender_bias_relative = gender_bias_time / total_time
            print(f"Gender bias evaluation time: {gender_bias_time:.2f} seconds ({gender_bias_relative:.2%} of total time)")
        print(
            f"Gender bias names evaluation time: {gender_bias_name_time:.2f} seconds ({gender_bias_name_relative:.2%} of total time)")

    accuracy = non_gender_mlm_stats['accuracy']
    for key in ['bpi', 'mpi', 'fpi']:
        score = gender_bias_name_stats[f'_{key}']
        result[key] = score * accuracy

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w+') as f:
        json.dump(result, f, indent=2)

    return result


def get_evaluation_file(path, gender_factors, lrs, thorough=True):
    iter_stats = lambda iterable: f'{min(iterable)}_{max(iterable)}_{len(iterable)}'
    not_thorough = '_not_thorough' if not thorough else ''
    return f'{path}_evaluation.json', f'{path}_evaluation_{iter_stats(gender_factors)}_{iter_stats(lrs)}{not_thorough}.json'

def evaluate_bert_with_ae(path, gender_factors=None, lrs=None, thorough=True, cache='../cache'):
    bert_with_ae = ModelWithGradiend.from_pretrained(path)
    base_model = bert_with_ae.bert
    tokenizer = bert_with_ae.tokenizer
    base_file, file = get_evaluation_file(path + f'/{cache}/decoder/' + os.path.basename(path), gender_factors, lrs, thorough=thorough)
    os.makedirs(os.path.dirname(base_file), exist_ok=True)
    os.makedirs(os.path.dirname(file), exist_ok=True)


    pairs = {(gender_factor, lr) for gender_factor in gender_factors for lr in lrs}
    expected_results = len(pairs) + 1 + 3 # 1 because of base, 3 because of bpi, mpi, fpi
    def convert_results_to_dict(list_results):
        dict_result = {}
        for entry in list_results:
            id = entry['id']
            if isinstance(id, str):
                key = id
            else:
                key = (id['gender_factor'], id['lr'])

            dict_result[key] = entry
        return dict_result

    def convert_results_to_list(dict_results):
        return [{**dict_result, 'id': (key if isinstance(key, str) else {'gender_factor': key[0], 'lr': key[1]})} for key, dict_result in dict_results.items()]

    try:
        relevant_results = json.load(open(file, 'r'))

        relevant_results = convert_results_to_dict(relevant_results)

        # check if complete
        if len(relevant_results) == expected_results:
            return relevant_results
    except FileNotFoundError:
        relevant_results = {}
    except Exception as e:
        print(f'Error for {file}')
        raise e

    try:
        all_results = json.load(open(base_file, 'r'))
        all_results = convert_results_to_dict(all_results)

        # copy relevant results into relevant_results
        for pair in pairs:
            if pair in all_results:
                relevant_results[pair] = all_results[pair]

        if 'base' in all_results:
            relevant_results['base'] = all_results['base']
        elif ('b', 'a') in all_results:
            # todo deprecated because of earlier error
            relevant_results['base'] = all_results[('b', 'a')]

        if len(relevant_results) == expected_results:
            with open(file, 'w+') as f:
                json.dump(convert_results_to_list(relevant_results), f, indent=2)
            return relevant_results

    except FileNotFoundError:
        all_results = {}

    if 'base' in relevant_results:
        print("Skipping base model as it is already evaluated")
    else:
        base_results = evaluate_model(base_model, tokenizer)
        all_results['base'] = base_results
        relevant_results['base'] = base_results


    for gender_factor, lr in tqdm(pairs, desc="Evaluate BERT With AE"):
        id = {'gender_factor': gender_factor, 'lr': lr}
        id_key = (gender_factor, lr)
        if id_key in relevant_results:
            print(f"Skipping {id} as it is already evaluated")
            continue

        enhanced_bert = bert_with_ae.modify_bert(lr=lr, gender_factor=gender_factor)
        enhanced_bert_results = evaluate_model(enhanced_bert, tokenizer, df_name=f'results/cache/models/{bert_with_ae.name}_lr_{lr}_gf_{gender_factor}.csv', thorough=thorough)
        all_results[id_key] = enhanced_bert_results
        relevant_results[id_key] = enhanced_bert_results

        with open(base_file, 'w+') as f:
            json.dump(convert_results_to_list(all_results), f, indent=2)

    raw_relevant_results = relevant_results.copy()
    for key in ['bpi', 'mpi', 'fpi']:
        print(relevant_results.items())
        arg_max = max(raw_relevant_results, key=lambda x: raw_relevant_results[x][key])
        if arg_max == 'base':
            gender_factor = 0
            lr = 0
        else:
            gender_factor = arg_max[0]
            lr = arg_max[1]
        relevant_results[key] = {
            'value': relevant_results[arg_max][key],
            'id': arg_max,
            'gender_factor': gender_factor,
            'lr': lr,
        }

    list_results = convert_results_to_list(relevant_results)
    with open(file, 'w+') as f:
        json.dump(list_results, f, indent=2)
    return convert_results_to_dict(list_results)

def plot_bert_with_ae_results(path, data, gender_factors, lrs, metrics=None, friend=True, split='total', thorough=True, highlight='best'):
    metrics = metrics or ['avg_prob_m', 'avg_prob_f', 'avg_prob_m + avg_prob_f', 'apd', 'accuracy', 'bpi', 'fpi', 'mpi']

    baseline = None
    evaluations = []
    for id, entry in data.items():
        if id == 'base':
            baseline = entry
        else:
            evaluations.append(entry)

    if baseline is None:
        print(json.dumps(data, indent=2))
        raise ValueError('No baseline found in data!')

    def get_metric(x, metric):
        bias = 'gender_bias_names' if friend else 'gender_bias'
        if metric == 'accuracy':
            x = x['mlm']['accuracy']
        elif metric in {'gender_preference_accuracy', 'overall_accuracy', 'average_absolute_difference'}:
            x = x['']
        elif metric == 'avg_prob_m + avg_prob_f':
            if isinstance(x[bias]['avg_prob_m'], dict):
                x = {k: xx + x[bias]['avg_prob_f'][k] for k, xx in x[bias]['avg_prob_m'].items()}
            else:
                x = x[bias]['avg_prob_m'] + x[bias]['avg_prob_f']
        elif metric in {'bpi', 'fpi', 'mpi'}:
            x = x[metric]
        else:
            x = x[bias][metric]

        if isinstance(x, float):
            return x
        return x[split]

        # Prepare the subplots

    n_metrics = len(metrics)
    if n_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))
        axes = [axes]
    elif n_metrics == 2 or n_metrics == 3:
        fig, axes = plt.subplots(1, n_metrics, figsize=(10 * n_metrics, 8))
    elif n_metrics == 4:
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
    elif n_metrics in [5, 6]:
        fig, axes = plt.subplots(2, 3, figsize=(30, 16))
        axes = axes.flatten()
    elif n_metrics in [7, 8]:
        fig, axes = plt.subplots(2, 4, figsize=(40, 16))
        axes = axes.flatten()
    else:
        raise ValueError(f"Too many metrics to plot: {n_metrics} ({metrics}")

    baseline_values = {metric: get_metric(baseline, metric) for metric in metrics}

    metrics_labels = {
        'apd': 'APD = Average Probability Difference = Sum(|P(M) - P(F)|)',
        'preference_score': 'Preference Score = |mean(P(M)) - mean(P(F))|',
        'avg_prob_f': 'Mean Probability of female token = mean(P(F))',
        'avg_prob_m': 'Mean Probability of male token = mean(P(M))',
        'he_prob': 'Mean for M has higher probability than F (P(M) > P(F))',
        'accuracy': 'MLM accuracy on non-gender text',
    }

    metric_labels_latex = {
        'apd': 'APD',
        'bpi': 'BPI',
        'fpi': 'FPI',
        'mpi': 'MPI',
        'avg_prob_f': r'\mathhbb{P}(F)',
        'avg_prob_m': r'\mathhbb{P}(M)',
        'avg_prob_m + avg_prob_f': r'\mathbb{P}(FM)',
        'accuracy': 'Accuracy',
    }

    y_tick_labels = [f'{factor:.1f}' for factor in gender_factors]

    metric_dfs = {}
    extreme_values = {}
    heatmap_data_dfs = {}
    masks = {}

    for metric in metrics:
        df = pd.DataFrame([{
            'gender_factor': e['id']['gender_factor'],
            'lr': e['id']['lr'],
            'value': get_metric(e, metric)
        } for e in evaluations if isinstance(e, dict) and isinstance(e['id'], dict)])
        metric_dfs[metric] = df

        # rename column gender_factor to "Gender Factor"
        df['Gender Factor'] = df['gender_factor']
        df['Learning Rate'] = df['lr']

        heatmap_data = df.pivot(index='Gender Factor', columns='Learning Rate', values='value')
        heatmap_data_dfs[metric] = heatmap_data
        baseline_value = baseline_values[metric]

        if metric in {'preference_score', 'apd'}:
            mask = heatmap_data < baseline_value
            extreme_value = heatmap_data.min().min()
        else:
            mask = heatmap_data > baseline_value
            extreme_value = heatmap_data.max().max()
        extreme_values[metric] = extreme_value
        masks[metric] = mask

    metric_colors = {
        'bpi': '#FF0000',
        'mpi': '#FFA500',
        'fpi': '#FF00FF',
    }
    tolerance = 0.01

    for ax, metric in zip(axes, metrics):
        metric_label = metrics_labels.get(metric, metric)
        heatmap_data = heatmap_data_dfs[metric]
        baseline_value = baseline_values[metric]
        mask = masks[metric]
        sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': ''}, annot=True, fmt='.2f',
                    annot_kws={"size": 8}, ax=ax, yticklabels=y_tick_labels)

        for y in range(heatmap_data.shape[0]):
            for x in range(heatmap_data.shape[1]):
                if highlight == 'best':
                    for metric, metric_color in metric_colors.items():
                        heatmap_data_metric = heatmap_data_dfs[metric]
                        extreme_value = extreme_values[metric]
                        if heatmap_data_metric.iloc[y, x] == extreme_value:
                            ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=metric_color, linewidth=3))
                else:
                    extreme_value = extreme_values[metric]
                    if abs(heatmap_data.iloc[y, x] - extreme_value) <= tolerance:
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', linewidth=3))
                    elif mask.iloc[y, x]:
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red'))

        ax.set_title(f"{metric_label}", fontsize=15)
        # Add a custom text for the baseline value with background color
        ax.text(1.0, 1.0, f"(Baseline: {baseline_value:.4f})",
                transform=ax.transAxes,  # Set relative to the axis
                fontsize=12,  # Font size
                ha='right',  # Horizontal alignment
                va='center',  # Vertical alignment
                bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))  # Background color

    if path.endswith('ae'):
        path = path.removesuffix('ae')
    model_name = os.path.basename(path)
    fig.suptitle(model_name, y=1, fontsize=25)
    plt.tight_layout(rect=[0, 0, 0.98, 1])

    def get_evaluation_file(path, gender_factors, lrs, thorough=True):
        iter_stats = lambda iterable: f'{min(iterable)}_{max(iterable)}_{len(iterable)}'
        not_thorough = '_not_thorough' if not thorough else ''
        return f'{path}_evaluation.json', f'{path}_evaluation_{iter_stats(gender_factors)}_{iter_stats(lrs)}{not_thorough}.json'

    _, image_file = get_evaluation_file(f'results/img/{model_name}', gender_factors, lrs, thorough=thorough)
    base_image_file = image_file.replace('.json', '.pdf')
    plt.savefig(base_image_file, bbox_inches='tight')
    plt.show()

    xticklabels = [f'{lr:.0e}' for lr in lrs]

    for idx, metric in enumerate(metrics):
        version = base_image_file.removesuffix(".pdf")
        heatmap_data = heatmap_data_dfs[metric]
        baseline_value = baseline_values[metric]
        mask = masks[metric]

        # Create a new figure for each subplot
        fig_single, ax_single = plt.subplots(figsize=(8, 5))


        # Plot the heatmap again for each subplot
        cbar_kws = {}
        if use_tex:
            cbar_kws = {'label': metric_labels_latex[metric]}
        sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws=cbar_kws, annot=True, fmt='.2f',
                    annot_kws={"size": 8}, ax=ax_single, yticklabels=y_tick_labels, xticklabels=xticklabels)

        for y in range(heatmap_data.shape[0]):
            for x in range(heatmap_data.shape[1]):
                if highlight == 'best':
                    for metric, metric_color in metric_colors.items():
                        heatmap_data_metric = heatmap_data_dfs[metric]
                        extreme_value = extreme_values[metric]
                        if heatmap_data_metric.iloc[y, x] == extreme_value:
                            ax_single.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor=metric_color, linewidth=3))
                else:
                    extreme_value = extreme_values[metric]
                    if abs(heatmap_data.iloc[y, x] - extreme_value) <= tolerance:
                        ax_single.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', linewidth=3))
                    elif mask.iloc[y, x]:
                        ax_single.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red'))

        # Determine the color based on the baseline_value and the cmap YlGnBu
        cmap = plt.get_cmap("YlGnBu")  # Get the colormap

        # Normalize baseline_value to a range between 0 and 1
        norm = matplotlib.colors.Normalize(vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max())

        # Map the baseline_value to the corresponding color in the colormap
        baseline_color = cmap(norm(baseline_value))

        # Function to compute luminance and decide text color
        def get_text_color(facecolor):
            # Convert color to RGB
            rgb = matplotlib.colors.to_rgb(facecolor)
            # Calculate luminance using relative luminance formula
            luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

            # If luminance is high (bright background), return black text, else return white text
            return 'black' if luminance > 0.5 else 'white'

        text_color = get_text_color(baseline_color)
        plt.title(f'Base Model: {baseline_value:.3f}', bbox={'facecolor': baseline_color, 'pad': 5}, color=text_color)

        ax_single.set_xlabel('Learning Rate', fontsize=12)
        ax_single.set_ylabel('Gender Factor', fontsize=12)
        # Save the individual subplot
        image_file = f'results/img/decoder/{model_name}/{version}/subplot_{metrics[idx]}.pdf'
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
        fig_single.savefig(image_file, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_single)  # Close to prevent memory issues

    return data


def default_evaluation(model, large=True, plot=True, cache='../cache', **kwargs):
    if large:
        gender_factors = [-100, -10, -5, -2] + list(np.linspace(-1, 1, 11)) + [2, 5, 10, 100]
        gender_factors = [-100, -10, -5, -2] + [-1, -0.5, 0.0, 0.5, 1.0] + [2, 5, 10, 100]
        gender_factors = [-10, -5, -2] + [-1, -0.5, 0.0, 0.5, 1.0] + [2, 5, 10]
        #gender_factors = [-10, -5, -2] + [-1, -0.75, -0.5, 0.25, 0.0, 0.25, 0.5, 0.75, 1.0] + [2, 5, 10]
        lrs = [-1e-1, -5e-2, -1e-2, -5e-3, -1e-3, -5e-4, -1e-4, -5e-5, -1e-5, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        lrs = [-5e-2, -1e-2, -5e-3, -1e-3, -5e-4, -1e-4, -5e-5, -1e-5, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    else:
        gender_factors = [-10, -1, 0, 1, 10]
        lrs = [-1e-1, -1e-2, -1e-3, -1e-4, -1e-5, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        gender_factors = [-10, 1, 0, 1, 10]
        gender_factors = [-1, 0, 0.5, 1]
        #gender_factors = [0,]
        lrs = [-0.0004, 0.0005, 0.005, 0.01]
        #lrs = [0.005]

    cache = cache + '/' + model

    data = evaluate_bert_with_ae(model, gender_factors, lrs, thorough=large, cache=cache)

    if plot:
        plot_bert_with_ae_results(model, data, gender_factors=gender_factors, lrs=lrs, thorough=large, **kwargs)

    return data

def evaluate_models():
    model1 = 'results/changed_models/bert-base-uncased/female'
    model2 = 'results/changed_models/bert-base-uncased/male'
    model5 = 'results/changed_models/bert-base-uncased-unbiased'
    model3 = 'results/changed_models/bert-large-cased/female'
    model4 = 'results/changed_models/bert-large-cased/male'
    model6 = 'results/changed_models/bert-large-cased-unbiased'
    models = [
        model1, model2, 'bert-base-uncased', 'bert-large-cased', model3, model4,model5, model6
    ]
    results = {}
    metrics = []
    for model_id in models:
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        result = evaluate_model(model, tokenizer, verbose=True)
        print(model_id)
        #print(json.dumps(result, indent=2))
        results[model_id] = result
        apd = result['gender_bias_names']['apd']['total']
        P_M = result['gender_bias_names']['avg_prob_m']['total']
        P_F = result['gender_bias_names']['avg_prob_f']['total']
        metrics.append((model_id, apd, P_M, P_F))

    print(json.dumps(metrics, indent=2))

def evaluate_gender_prediction_metrics(results_df):
    # Initialize a dictionary to store metrics
    metrics = {}

    # Get all unique splits from the DataFrame
    splits = results_df['split'].unique()

    # Calculate accuracy for each split
    for split in splits:
        split_df = results_df[results_df['split'] == split]
        accuracy = split_df['correct'].mean()  # Mean of correct predictions (True = 1, False = 0)
        male_accuracy = split_df[split_df['true_gender'] == 'M']['correct'].mean()
        female_accuracy = split_df[split_df['true_gender'] == 'F']['correct'].mean()

        metrics[split] = {
            'accuracy': accuracy,
            'male_accuracy': male_accuracy,
            'female_accuracy': female_accuracy,
            #'total_samples': len(split_df),
            #'correct_predictions': split_df['correct'].sum(),
            'male_prob_mean': split_df['male_prob'].mean(),
            'female_prob_mean': split_df['female_prob'].mean(),
        }

    # Calculate overall accuracy
    overall_accuracy = results_df['correct'].mean()
    male_accuracy = results_df[results_df['true_gender'] == 'M']['correct'].mean()
    female_accuracy = results_df[results_df['true_gender'] == 'F']['correct'].mean()

    metrics['total'] = {
        'accuracy': overall_accuracy,
        'male_accuracy': male_accuracy,
        'female_accuracy': female_accuracy,
        #'total_samples': len(results_df),
        #'correct_predictions': results_df['correct'].sum(),
        'male_prob_mean': results_df['male_prob'].mean(),
        'female_prob_mean': results_df['female_prob'].mean()
    }

    df = pd.DataFrame(metrics).T
    # add the index as column "split"
    df['split'] = df.index
    return df

def evaluate_gender_prediction(model_name, target=None):
    assert target in {None, 'M', 'F'}

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return evaluate_gender_prediction_by_model(model, tokenizer, target=target)


def evaluate_gender_prediction_by_model(model, tokenizer, target=None, target_words=('he', 'she')):
    model.eval()
    names = read_namexact()

    # Convert male and female tokens to IDs once
    target_word_M, target_word_F = target_words

    male_token_id = tokenizer.convert_tokens_to_ids(target_word_M)
    female_token_id = tokenizer.convert_tokens_to_ids(target_word_F)

    results = []

    for split, split_names in names.groupby('split'):

        split_data = []
        genders = []
        names_list = []

        # Collect all the text and gender info for this split
        for _, (name, gender) in split_names[['name', 'gender']].iterrows():
            text = f'Is {name} a {target_word_M} or a {target_word_F}? {name} is a [MASK].'

            #text = f'The individual named {name} is [MASK] by gender.'
            split_data.append(text)
            genders.append(gender)
            names_list.append(name)

        # Tokenize the entire batch of text at once
        tokenized_text = tokenizer(split_data, return_tensors="pt", padding=True, truncation=True)
        tokenized_text = {k: v.to(model.device) for k, v in tokenized_text.items()}

        with torch.no_grad():
            # Get model outputs (logits) for the batch
            outputs = model(**tokenized_text)
            logits = outputs.logits

        # Get the index of the [MASK] token
        mask_token_index = (tokenized_text['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)

        # Extract logits for the [MASK] token positions
        mask_logits = logits[mask_token_index]

        # Calculate probabilities for male and female tokens using softmax over all predictions
        probabilities = torch.softmax(mask_logits, dim=-1)

        # Extract male and female probabilities for each name in the batch
        male_probs = probabilities[:, male_token_id]
        female_probs = probabilities[:, female_token_id]

        # Process the results for each item in the batch
        for i, (name, gender, male_prob, female_prob) in enumerate(zip(names_list, genders, male_probs, female_probs)):
            # Predicted gender based on the highest probability
            predicted_gender = "M" if male_prob > female_prob else "F"
            if target is None:
                correct_prediction = (predicted_gender == gender)
            elif target == 'M':
                correct_prediction = (predicted_gender == 'M')
            elif target == 'F':
                correct_prediction = (predicted_gender == 'F')
            else:
                raise ValueError('Invalid target', target)

            # Collect the result for this name
            results.append({
                'name': name,
                'true_gender': gender,
                'predicted_gender': predicted_gender,
                'correct': correct_prediction,
                'male_prob': male_prob.item(),
                'female_prob': female_prob.item(),
                'split': split
            })

    # Return the results as a DataFrame
    df = pd.DataFrame(results)

    metrics = evaluate_gender_prediction_metrics(df)

    print(metrics.to_string())

    return metrics
