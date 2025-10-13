from gradiend.setups.util.multiclass import MultiClassSetup, create_training_dataset, Bias3DCombinedSetup
from gradiend.training.gradiend_training import train_for_configs


class Bias1DSetup(MultiClassSetup):
    def __init__(self, bias_type, class1, class2, pretty_id=None):
        super().__init__(bias_type, class1, class2, pretty_id=pretty_id)
        self.n_features = 1

    @property
    def class1(self):
        return self.classes[0]

    @property
    def class2(self):
        return self.classes[1]

class Race1DSetup(Bias1DSetup):
    def __init__(self, class1, class2, pretty_id=None):
        super().__init__('race', class1, class2, pretty_id=pretty_id)

class WhiteBlackSetup(Race1DSetup):
    def __init__(self):
        super().__init__('white', 'black', pretty_id='Black/White')


class WhiteAsianSetup(Race1DSetup):
    def __init__(self):
        super().__init__('white', 'asian', pretty_id='Asian/White')

class BlackAsianSetup(Race1DSetup):
    def __init__(self):
        super().__init__('black', 'asian', pretty_id='Asian/Black')

class Religion1DSetup(Bias1DSetup):
    def __init__(self, religion1, religion2, pretty_id=None):
        super().__init__('religion', religion1, religion2, pretty_id=pretty_id)

class MuslimJewishSetup(Religion1DSetup):
    def __init__(self):
        super().__init__('muslim', 'jewish', pretty_id='Jewish/Muslim')

class ChristianJewishSetup(Religion1DSetup):
    def __init__(self):
        super().__init__('christian', 'jewish', pretty_id='Christian/Jewish')

class ChristianMuslimSetup(Religion1DSetup):
    def __init__(self):
        super().__init__('christian', 'muslim', pretty_id='Christian/Muslim')



class Race3DCombinedSetup(Bias3DCombinedSetup):
    def __init__(self):
        super().__init__('race', 'white', 'black', 'asian')

        self.init_gradiends = [
            f'results/models/{self.bias_type}_white_black',
            f'results/models/{self.bias_type}_white_asian',
            f'results/models/{self.bias_type}_black_asian',
        ]

class Religion3DCombinedSetup(Bias3DCombinedSetup):
    def __init__(self):
        super().__init__('religion', 'christian', 'muslim', 'jewish')

        self.init_gradiends = [
            f'results/models/{self.bias_type}_christian_jewish',
            f'results/models/{self.bias_type}_christian_muslim',
            f'results/models/{self.bias_type}_muslim_jewish',
        ]

class Race3DSetup(MultiClassSetup):
    def __init__(self):
        super().__init__('race', 'white', 'black', 'asian')
        self.n_features = 3


    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(classes=self.classes, *args, **kwargs)



def train_multi_dim_gradiends(configs, version=None, activation='tanh', setups=None):
    for id, config in configs.items():
        config['activation'] = activation
        config['delete_models'] = False

    setups = setups or [WhiteBlackSetup(), WhiteAsianSetup(), BlackAsianSetup()]

    for setup in setups:
        print(f"Training setup: {setup.id}")
        #if isinstance(setup, Bias3DCombinedSetup):
        #    configs = {k: {**v, 'max_iterations': 500} for k, v in configs.items()}

        train_for_configs(setup, configs, version=version, n=1, clear_cache=True)


if __name__ == '__main__':
    configs = {
        'bert-base-cased': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=8),
        'bert-large-cased': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=4),
        'distilbert-base-cased': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff'),
        'roberta-large': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=4),

        'gpt2': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff'),
        'meta-llama/Llama-3.2-3B-Instruct': dict(eval_max_size=50, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=1, torch_dtype=torch.bfloat16, lr=1e-4),
        'meta-llama/Llama-3.2-3B': dict(eval_max_size=50, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=1, torch_dtype=torch.bfloat16, lr=1e-4),
    }

    configs_factual_source = {k: {**v, 'source': 'factual'} for k, v in configs.items()}

    all_setups = [
        ChristianJewishSetup(),
        MuslimJewishSetup(),
        WhiteAsianSetup(),
        BlackAsianSetup(),
        WhiteBlackSetup(),
        ChristianMuslimSetup(),
    ]

    try:
        train_multi_dim_gradiends(configs, version='v5', activation='tanh', setups=all_setups)
    except NotImplementedError as e:
        print(f"Error during training: {e}")
