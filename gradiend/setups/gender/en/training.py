import numpy as np

from gradiend.setups import Setup
from gradiend.setups.emotion import plot_encoded_by_class
from gradiend.setups.gender.en import GenderEnSetup, create_training_dataset
from gradiend.training.gradiend_training import train_for_configs, train as train_gradiend


class MultiDimGenderEnSetup(Setup):

    def __init__(self, n_features=2):
        super().__init__('gender-en', n_features=n_features)

    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(*args, **kwargs)

    def evaluate(self, model_with_gradiend, eval_data, eval_batch_size=32, config=None, training_stats=None):
        # one hot encode the labels
        #if isinstance(eval_data['labels'][0], int):
        #    num_classes = max(eval_data['labels']) + 1
        #    eval_data['labels'] = np.eye(num_classes)[eval_data['labels']]

        result = super().evaluate(model_with_gradiend, eval_data, eval_batch_size=eval_batch_size)
        score = result['score']
        encoded = result['encoded']
        encoded_by_class = result['encoded_by_class']
        mean_by_class = result['mean_by_class']

        output_name = f'training_{str(model_with_gradiend.gradiend.encoder[1])}.pdf'
        if config and 'output' in config:
            base_output = config['output']
            global_step = training_stats.get('global_step', None)
            output = f'{base_output}/{global_step}_{output_name}'
        else:
            output = f'img/{output_name}'
        plot_encoded_by_class(encoded_by_class, mean_by_class=mean_by_class, title=f"Score {score}", output=output)

        return result


def default_training(configs):
    setup = GenderEnSetup()
    train_for_configs(setup, configs, n=1)

def dual_training(configs, version=None, activation='tanh'):
    setup = MultiDimGenderEnSetup(n_features=2)
    for id, config in configs.items():
        config['activation'] = activation
        config['delete_models'] = True
    train_for_configs(setup, configs, version=version, n=3)



def multi_dim_training(configs, version=None, activation='tanh', n_features=2):
    setup = MultiDimGenderEnSetup(n_features=n_features)
    for id, config in configs.items():
        config['activation'] = activation
        config['delete_models'] = True
    train_for_configs(setup, configs, version=version, n=2)



if __name__ == '__main__':
    model_configs = {
        'bert-base-cased': dict(),
        'bert-large-cased': dict(eval_max_size=0.5, eval_batch_size=4),
        'distilbert-base-cased': dict(),
        'roberta-large': dict(eval_max_size=0.5, eval_batch_size=4),
        'gpt2': dict(),
        'meta-llama/Llama-3.2-3B-Instruct': dict(batch_size=32, eval_max_size=0.05, eval_batch_size=1, epochs=1, torch_dtype=torch.bfloat16, lr=1e-4),
        'meta-llama/Llama-3.2-3B': dict(batch_size=32, eval_max_size=0.05, eval_batch_size=1, epochs=1, torch_dtype=torch.bfloat16, lr=1e-4, n_evaluation=250),
    }

    setup = GenderEnSetup()

    models = []
    for base_model, model_config in model_configs.items():
        model = train_gradiend(setup, base_model, model_config, n=3, version='', clear_cache=False, force=False)
        models.append(model)

    for model in models:
        setup.select(model)
