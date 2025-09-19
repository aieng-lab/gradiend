from pprint import pprint

import numpy as np
import torch
import json
import os

from gradiend.evaluation.analyze_encoder import analyze_models
from gradiend.training.gradiend_training import train_all_layers_gradiend


def run(model, seeds=tuple(range(10)), n=None, eval_batch_size=None, eval_max_size=100, output_version='', n_eval=None, lr=1e-5):
    converged_labels = []
    convergence_classes = []
    mean_values = []
    if n is None:
        if model == 'bert-base-uncased' or model == 'bert-large-cased':
            n = 500
        else:
            n = 100

    n_eval = n_eval or n


    if eval_batch_size is None:
        if 'large' in model:
            eval_batch_size = 4
        else:
            eval_batch_size = 32
    base_output = f'results/experiments/convergence/{model}'
    if output_version:
        base_output += f'/{output_version}'

    for seed in seeds:
        output_file = f'{base_output}/{seed}/results.json'

        if os.path.isfile(output_file):
            print(f'File {output_file} already exists, skipping')
            continue

        config = {
            'model': model,
            'epochs': 1,
            'max_iterations': n,
            'n_evaluation': n_eval,
            'eval_max_size': eval_max_size,
            'seed': seed,
            'eval_batch_size': eval_batch_size,
            'lr': lr,
        }

        if 'llama' in model.lower():
            config['torch_dtype'] = torch.bfloat16


        output = f'{base_output}/{seed}'
        print('Training', output)
        train_all_layers_gradiend(output=output, **config)

        df = analyze_models(output, max_size=100)

        df_genter = df[df['type'] == 'gender masked']
        df_gender_neutral = df[df['type'] != 'gender masked']
        mean_male_encoded = df_genter[df_genter['state'] == 'M']['encoded'].mean().item()
        mean_female_encoded = df_genter[df_genter['state'] == 'F']['encoded'].mean().item()
        mean_gender_neutral_encoded = df_gender_neutral['encoded'].mean().item()
        print('Male', mean_male_encoded, 'Female', mean_female_encoded, 'Gender Neutral', mean_gender_neutral_encoded)

        converged = False
        if mean_male_encoded < -0.5 and mean_female_encoded > 0.5 and abs(mean_gender_neutral_encoded) < 0.5:
            converged = True
            convergence_classes.append(1)
        elif mean_male_encoded > 0.5 and mean_female_encoded > 0.5:
            convergence_classes.append(2)
        elif mean_male_encoded < -0.5 and mean_female_encoded < -0.5:
            convergence_classes.append(3)
        else:
            convergence_classes.append(4)
        converged_labels.append(converged)
        mean_values.append((mean_male_encoded, mean_female_encoded, mean_gender_neutral_encoded))

        # Save the results as json
        output = {
            'converged': converged,
            'convergence_classes': convergence_classes,
            'mean_values': mean_values,
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=4)

        # free memory
        del df
        torch.cuda.empty_cache()
        # garbage collection
        import gc
        gc.collect()



    print('Converged:', np.mean(converged_labels))
    print('Convergence classes:', convergence_classes)

    pprint(mean_values)

    # Save the results as json
    output_file = f'{base_output}/results.json'
    output = {
        'converged': np.mean(converged_labels).item(),
        'convergence_classes': convergence_classes,
        'mean_values': mean_values,
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)

    config_hash = hash(frozenset(config.items()))
    output_hash_file = f'{base_output}/results_{config_hash}.json'
    with open(output_hash_file, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    #run('bert-base-uncased')
    #run('bert-large-cased')
    #run('roberta-base')
    #run('distilbert-base-cased')
    #run('roberta-large')

# lr5 version uses lr4!!
    run('meta-llama/Llama-3.2-3B-Instruct', seeds=tuple(range(10)), n=500, n_eval=500, eval_batch_size=1, eval_max_size=10, lr=1e-4, output_version='lr4')
    #run('meta-llama/Llama-3.2-3B', seeds=tuple(range(10)), n=1000, eval_batch_size=1, eval_max_size=10, lr=1e-5, output_version='lr5')