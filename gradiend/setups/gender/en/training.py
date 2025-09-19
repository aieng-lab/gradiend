import numpy as np

from gradiend.setups import Setup
from gradiend.setups.emotion import plot_encoded_by_class
from gradiend.setups.gender.en import GenderEnSetup, create_training_dataset
from gradiend.training.gradiend_training import train_for_configs


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
    configs = {
        #'distilbert-base-cased': dict(),
        'results/decoder-mlm-head/gpt2': dict(eval_max_size=None, eval_batch_size=4),
    }

    default_training(configs)

    exit(1)

    #dual_training(configs, version='tanh', activation='tanh')
    #dual_training(configs, version='relu', activation='relu')
    #dual_training(configs, version='sigmoid', activation='sigmoid')
    multi_dim_training(configs, version='gelu_3', activation='gelu', n_features=3)
    multi_dim_training(configs, version='tanh_3', activation='tanh', n_features=3)
    multi_dim_training(configs, version='tanh_5', activation='tanh', n_features=5)
    multi_dim_training(configs, version='gelu_5', activation='gelu', n_features=5)
    #dual_training(configs, version='silu', activation='silu')
    #dual_training(configs, version='smht', activation='smht')
    #dual_training(configs, version='id', activation='id')
