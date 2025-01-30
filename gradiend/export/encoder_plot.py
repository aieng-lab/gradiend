from gradiend.evaluation.analyze_encoder import plot_encoded_value_distribution, analyze_models
from gradiend.export import models
from gradiend.util import init_matplotlib

if __name__ == '__main__':
    names = list(models.values())
    models = list(models.keys())

    suffix = ''
    gradiend_models = [f'results/models/{model}{suffix}' for model in models]

    init_matplotlib(use_tex=True)

    # make sure that the models have been analyzed before
    analyze_models(*gradiend_models)

    plot_encoded_value_distribution(*gradiend_models, model_names=names)