import json
import os
import shutil
import time

import numpy as np

from gradiend.evaluation.analyze_encoder import analyze_models, get_model_metrics
from gradiend.evaluation.select_models import select
from gradiend.training import train_all_layers_gradiend, train_multiple_layers_gradiend


def train(base_model, model_config, n=3, metric='pearson_MF', force=False, version=None, clear_cache=False):
    metrics = []
    total_start = time.time()
    times = []
    if version is None:
        version = ''
    else:
        version = f'/v{version}'

    for i in range(n):
        start = time.time()
        output = f'results/experiments/gradiend/{base_model}{version}/{i}'
        metrics_file = f'{output}/metrics.json'
        if not force and os.path.exists(metrics_file):
            metrics.append(json.load(open(metrics_file)))
            print(f'Skipping training of {output} as it already exists')
            continue

        if not os.path.exists(output):
            print('Training', output)
            model_config['seed'] = i+1
            if 'layers' in model_config:
                train_multiple_layers_gradiend(model=base_model, output=output, **model_config)
            else:
                train_all_layers_gradiend(model=base_model, output=output, **model_config)
        else:
            print('Model', output, 'already exists, skipping training, but evaluate')

        analyze_models(output, split='val', force=force)
        model_metrics = get_model_metrics(output, split='val')
        metric_value = model_metrics[metric]
        json.dump(metric_value, open(metrics_file, 'w'))
        metrics.append(metric_value)

        times.append(time.time() - start)

        if clear_cache:
            cache_folder = f'data/cache/gradients/{base_model}'
            if os.path.exists(cache_folder):
                shutil.rmtree(cache_folder)

    print(f'Metrics for model {base_model}: {metrics}')
    best_index = np.argmax(metrics)
    print('Best metric at index', best_index, 'with value', metrics[best_index])

    output = f'results/models/{base_model}{version.replace("/", "-")}'
    # copy the best model to output
    shutil.copytree(f'results/experiments/gradiend/{base_model}{version}/{best_index}', output, dirs_exist_ok=True)

    total_time = time.time() - total_start
    if times:
        print(f'Trained {len(times)} models in {total_time}s')
        print(f'Average time per model: {np.mean(times)}')
    else:
        print('All models were already trained before!')

    return output


if __name__ == '__main__':
    model_configs = {
        'bert-base-cased': dict(),
        'bert-large-cased': dict(eval_max_size=0.5, eval_batch_size=4),
        'distilbert-base-cased': dict(),
        'roberta-large': dict(eval_max_size=0.5, eval_batch_size=4),
        #'answerdotai/ModernBERT-base': dict(), # ModernBert not working with current transformers version!
        #'gpt2': dict(),
        #'gpt2-medium': dict(),
        #'gpt2-large': dict(),
        #'gpt2-xl': dict(layers=['*.h.47.*'], eval_max_size=0.1, eval_batch_size=4),
    }

    models = []
    for base_model, model_config in model_configs.items():
        model = train(base_model, model_config, n=3, version='', clear_cache=False)
        models.append(model)

    for model in models:
        select(model)
