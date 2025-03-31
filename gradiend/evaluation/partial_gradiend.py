import json
import os

import numpy as np
from matplotlib import pyplot as plt
from transformers import AutoModel

from gradiend.evaluation.analyze_decoder import default_evaluation
from gradiend.model import ModelWithGradiend
from gradiend.util import convert_tuple_keys_to_strings, convert_string_keys_to_tuples

proportions = np.arange(0, 1.1, 0.1)
proportions = [0.0, 1e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
#proportions = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
proportions = [1e-7, 1e-5, 1e-3, 1e-2, 1e-1]
#proportions = [1e-6]
#proportions = [0.0, 1e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5]

#proportions = [1.0]

print(proportions)

def evaluate(model, part, top_k_part):
    #model = 'results/models/bert-base-cased'
    #model = 'results/models/bert-large-cased'
    output = 'results/experiments/gradiend_proportions'
    model_id = model.split("/")[-1]
    results = {}
    #part = 'decoder'
    #top_k_part = 'decoder-bias'
    #top_k_part = 'decoder'
    #top_k_part = 'decoder-sum'
    part_str = f'_part_{part}' if part != 'decoder' else ''
    top_k_part_str = f'_top_k_part_{top_k_part}' if top_k_part != 'decoder' else ''
    output_file = f'{output}/{model_id}/eval_{min(proportions)}_{max(proportions)}_{len(proportions)}{part_str}{top_k_part_str}.json'

    base_model = AutoModel.from_pretrained(model)
    model_with_gradiend = ModelWithGradiend.from_pretrained(model)
    # n is number of total parameters of base_model
    n = sum(p.numel() for p in base_model.parameters())

    if os.path.isfile(output_file):
        json_results = json.load(open(output_file))
        results = convert_string_keys_to_tuples(json_results)
        results = {float(k): v for k, v in results.items()}
    else:
        for prop in proportions:
            eval = default_evaluation(model_with_gradiend, top_k=prop, plot=False, top_k_part=top_k_part)
            n_neurons = int(n * prop)
            eval['n_neurons'] = n_neurons
            results[prop] = eval
            print(f'{prop}: N=({n_neurons})')
            print(eval['bpi'])

        json_results = convert_tuple_keys_to_strings(results)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(json_results, f)

        # todo recalculate first 10 pairs of lr=0.1

    fpis = [results[prop]['fpi']['value'] for prop in proportions]
    mpis = [results[prop]['mpi']['value'] for prop in proportions]
    bpis = [results[prop]['bpi']['value'] for prop in proportions]

    print('FPI:', fpis)
    print('MPI:', mpis)
    print('BPI:', bpis)

    base_mpi = results[proportions[0]]['base']['mpi']
    base_fpi = results[proportions[0]]['base']['fpi']
    base_bpi = results[proportions[0]]['base']['bpi']

    plt.plot(proportions, fpis, label='FPI', color='orange')
    plt.plot(proportions, mpis, label='MPI', color='green')
    plt.plot(proportions, bpis, label='BPI', color='blue')

    # plot base values as constant lines
    #plt.axhline(base_fpi, color='orange', linestyle='--', label='Base FPI')
    #plt.axhline(base_mpi, color='green', linestyle='--', label='Base MPI')
    #plt.axhline(base_bpi, color='blue', linestyle='--', label='Base BPI')

    # label x axis with both proportions and number of neurons
    ax = plt.gca()  # Get the current axis
    ax.set_xticks(proportions)  # Force tick positions
    ax.set_xticklabels([f'{prop:.1e} (N={int(n * prop)})' for prop in proportions])  # Force labels
    plt.xscale('log')  # Set log scale after ensuring ticks

    plt.legend()
    plt.xlabel('Proportion/ Number of neurons')
    plt.ylabel('Accuracy')
    plt.grid()
    output = f'results/img/partial_gradiend/{model_id}_{part}_{top_k_part}.pdf'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.title(f'{model.split("/")[-1]} {part} {top_k_part}')
    plt.savefig(output)
    plt.show()

    plt.close()


    for prop in proportions:
        print(f'{prop}: N=({int(n * prop)})')

if __name__ == '__main__':

    models = [
        'results/models/distilbert-base-cased',
        #'results/models/bert-base-cased',
        'results/models/bert-large-cased',
        'results/models/roberta-large',
    ]

    top_k_parts = ['decoder', 'decoder-bias', 'decoder-sum']
    top_k_parts = ['decoder-bias']
    parts = ['decoder']

    for model in models:
        for part in parts:
            for top_k_part in top_k_parts:
                evaluate(model, part, top_k_part)
