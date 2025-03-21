from gradiend.evaluation.partial_gradiend_heuristic import get_heuristic_mask
from gradiend.evaluation.select_models import select
from gradiend.model import ModelWithGradiend
from gradiend.training.gradiend_training import train

models = [
    'bert-base-cased'
]

# trains a partial gradiend model with a heuristic mask based on averaged gradients based on a few samples
def train_heuristic_partial_gradiends(heuristic_threshold=1e-2):

    model_configs = {
        'bert-base-cased': dict(),
    }

    models = []
    for base_model, model_config in model_configs.items():
        print('Training', base_model)
        heuristic_layers, heuristic_n_neurons = get_heuristic_mask(base_model, heuristic_threshold)
        model_config['layers'] = heuristic_layers
        model = train(base_model, model_config, n=1, version=f'_{heuristic_threshold}', clear_cache=False)
        models.append(model)
        print('Trained', base_model)

    for model in models:
        select(model)


# trains the partial gradiend model with the best possible heuristic (based on a full gradiend model)
def train_best_heuristic_partial_gradiends(top_k=1e-5, top_k_part='decoder'):
    model_configs = {
         'bert-base-cased': dict(),
        # 'bert-large-cased': dict(eval_max_size=0.5, eval_batch_size=4),
        # 'distilbert-base-cased': dict(),
        # 'roberta-large': dict(eval_max_size=0.5, eval_batch_size=4),
        # 'answerdotai/ModernBERT-base': dict(), # ModernBert not working with current transformers version!
        # 'gpt2': dict(),
        # 'gpt2-medium': dict(),
        # 'gpt2-large': dict(),
        # 'gpt2-xl': dict(layers=['*.h.47.*'], eval_max_size=0.1, eval_batch_size=4),
    }

    models = []
    for base_model, model_config in model_configs.items():
        print('Training', base_model)
        gradiend = ModelWithGradiend.from_pretrained(f'results/models/{base_model}')
        #heuristic_layers = get_heuristic_mask(gradiend, top_k=top_k, top_k_part=top_k_part, part=part)
        heuristic_layers = gradiend.get_layer_mask(top_k, part=top_k_part)
        model_config['layers'] = heuristic_layers
        model = train(base_model, model_config, n=1, version=f'_{top_k}_{top_k_part}', clear_cache=False)
        models.append(model)
        print('Trained', base_model)

    for model in models:
        select(model)



if __name__ == '__main__':
    top_k_parts = ['decoder-bias'] # , 'decoder-sum', 'decoder',
    top_k_map = {
        'decoder-bias': 1e-7,
        #'decoder': 1e-5,
    }

    for top_k_part in top_k_parts:
        #train_heuristic_partial_gradiends()
        train_best_heuristic_partial_gradiends(top_k_part=top_k_part, top_k=top_k_map[top_k_part])