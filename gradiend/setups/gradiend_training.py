from gradiend.setups.gender.en import GenderEnSetup
from gradiend.training.gradiend_training import train_for_configs


def train(setup, configs):
    train_for_configs(setup, configs)


if __name__ == '__main__':
    setup = GenderEnSetup()

    configs = {
        'bert-base-cased': dict()
    }

    train(setup, configs)