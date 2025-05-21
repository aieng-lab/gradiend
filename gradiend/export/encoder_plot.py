from gradiend.evaluation.analyze_encoder import plot_encoded_value_distribution, analyze_models
from gradiend.export import models as default_models
from gradiend.util import init_matplotlib



def plot(models=None, suffix=''):
    if not models:
        models = default_models
        models = {f'results/models/{model}{suffix}': name for model, name in models.items()}
    elif isinstance(models, str):
        models = {models: models}

    names = list(models.values())
    gradiend_models = list(models.keys())


    init_matplotlib(use_tex=True)

    # make sure that the models have been analyzed before
    analyze_models(*gradiend_models)

    plot_encoded_value_distribution(*gradiend_models, model_names=names)


if __name__ == '__main__':
    plot()