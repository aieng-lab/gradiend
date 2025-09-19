import json
import pickle

from gradiend.export import models as pretty_models
from gradiend.model import ModelWithGradiend
from gradiend.setups.race.training import WhiteBlackSetup, BlackAsianSetup, WhiteAsianSetup, ChristianJewishSetup, \
    ChristianMuslimSetup, MuslimJewishSetup

base_models = [
    'roberta-large-v7',
    'bert-base-cased-v7',
    'distilbert-base-cased-v7',
    'gpt2-v7',
    'bert-large-cased-v7',
    'Llama-3.2-3B-v5',
    'Llama-3.2-3B-Instruct-v5',
]



setups = [
    BlackAsianSetup(),
    WhiteAsianSetup(),
    WhiteBlackSetup(),
    ChristianJewishSetup(),
    ChristianMuslimSetup(),
    MuslimJewishSetup(),
]

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=int, default=0, help='Setup type: 0=BlackAsian, 1=WhiteAsian, 2=WhiteBlack, 3=ChristianJewish, 4=ChristianMuslim, 5=MuslimJewish')
    args = parser.parse_args()
    return args

args = parse_args()

setup = setups[int(args.type)]
print(f'Using setup {setup.id}')
#setups = [setup]

setup_stats_cache_file = f'results/models/selection_race_v7.pkl'
try:
    setup_stats = pickle.load(open(setup_stats_cache_file, 'rb'))
except FileNotFoundError:
    setup_stats = {}

for setup in setups:
    for model in base_models:
        key = f'{setup.id}_{model}'
        if key in setup_stats:
            print(f'Skipping {key}, already computed')
            continue
        model_id = f'results/models/{setup.id}/{model}'
        #setup.select(model_id)
        model = ModelWithGradiend.from_pretrained(model_id)
        setup.plot_model_selection(model)

        stats = setup.get_model_selection_stats(model)
        setup_stats[key] = stats

pickle.dump(setup_stats, open(setup_stats_cache_file, 'wb'))

print()

# print selected model table

header = ['Model', r'FF $h$', r'LR $\alpha$', r'Base \P(Asian)', r'\P(Asian)', r'Base \P(Black)', r'\P(Black)', r'Base \P(White)', r'\P(White)', r'\accdec', r'\bpi']

for setup in setups:

    print(r'\midrule')
    print(rf'\multicolumn{{9}}{{c}}{{$\gradiend_{{{setup.pretty_id}}}$}} \\')
    print(r'\midrule')

    races = setup.races
    c1, c2 = races
    swapped = False
    if c1 > c2:
        swapped = True

    for model in base_models:
        stats = setup_stats.get(f'{setup.id}_{model}', None)
        if stats is None:
            continue

        base_stats = stats['base']
        best_stats = list(sorted(stats.values(), key=lambda x: x[setup.id]))[-1]

        lr = best_stats['id']['lr'] * (-1 if swapped else 1)
        ff = best_stats['id']['feature_factor']
        lms = best_stats['lms']['lms']
        bias_stats = base_stats[f'bias_{setup.id}']

        if 'race' in setup.id:
            prob1 = bias_stats['group_probs'].get('asian', None)
            prob2 = bias_stats['group_probs'].get('black', None)
            prob3 = bias_stats['group_probs'].get('white', None)

            base_prob1 = base_stats[f'bias_{setup.id}']['group_probs'].get('asian', None)
            base_prob2 = base_stats[f'bias_{setup.id}']['group_probs'].get('black', None)
            base_prob3 = base_stats[f'bias_{setup.id}']['group_probs'].get('white', None)
        else:
            prob1 = bias_stats['group_probs'].get('christian', None)
            prob2 = bias_stats['group_probs'].get('jewish', None)
            prob3 = bias_stats['group_probs'].get('muslim', None)

            base_prob1 = base_stats[f'bias_{setup.id}']['group_probs'].get('christian', None)
            base_prob2 = base_stats[f'bias_{setup.id}']['group_probs'].get('jewish', None)
            base_prob3 = base_stats[f'bias_{setup.id}']['group_probs'].get('muslim', None)


        score = best_stats[setup.id]
        pretty_model = pretty_models[model.removesuffix('-v5').removesuffix('-v7')]
        row = [f'{pretty_model}']


        not_none_prob1 = prob1 or prob2
        not_none_prob2 = prob2 or prob3
        not_none_base_prob1 = base_prob1 or base_prob2
        not_none_base_prob2 = base_prob2 or base_prob3


        row.append(f'{ff:.1f}')
        row.append(f'{lr:.1f}')
        fmt_probs = '.1e'
        row.append(f'{not_none_base_prob1:{fmt_probs}}' if not_none_base_prob1 is not None else "--")
        row.append(f'{not_none_prob1:{fmt_probs}}' if not_none_prob1 is not None else "--")
        row.append(f'{not_none_base_prob2:{fmt_probs}}' if not_none_base_prob2 is not None else "--")
        row.append(f'{not_none_prob2:{fmt_probs}}' if not_none_prob2 is not None else "--")
        row.append(f'{lms:.3f}')
        row.append(f'{score:{fmt_probs}}')

        row = [str(x) if not isinstance(x, float) else f'{x:.3f}' for x in row]

        row = [r.replace("e-0", "e-").replace("e+0", "e+") for r in row]

        print(' & '.join(row) + r' \\')

