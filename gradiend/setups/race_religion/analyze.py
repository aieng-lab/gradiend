import pandas as pd

from gradiend.model import ModelWithGradiend
from gradiend.setups.race_religion.training import WhiteBlackSetup, WhiteAsianSetup, BlackAsianSetup, ChristianJewishSetup, \
    ChristianMuslimSetup, MuslimJewishSetup
from gradiend.util import init_matplotlib

init_matplotlib(use_tex=True)



def analyze(setup, model_id):
    model = ModelWithGradiend.from_pretrained(model_id)
    output = f'{model_id}/encoded_values.csv'
    setup.analyze_model(model, output)
    print("Analysis complete.")
    setup.get_model_metrics(output, plot=False)


if __name__ == '__main__':
    setup = WhiteBlackSetup()

    base_models = [
        'bert-base-cased',
        'bert-large-cased',
        'distilbert-base-cased',
        'roberta-large',
        'gpt2',
        'Llama-3.2-3B',
        'Llama-3.2-3B-Instruct',
    ]

    setups = [WhiteBlackSetup(), WhiteAsianSetup(), BlackAsianSetup(), ChristianJewishSetup(), ChristianMuslimSetup(), MuslimJewishSetup()]

    for _ in range(2): # we call each analyze script twice as the first call might end in throwing an exception after normalizing the data; already analyzed data is cached, i.e., there is no computational overhead
        for base_model in base_models:
            for setup in setups:
                try:
                    print(f'Analyzing {setup.id} with base model {base_model}')
                    model = f'results/models/{setup.id}/{base_model}-v7'
                    analyze(setup, model)

                except Exception as e:
                    print(f"Error analyzing {base_model}: {e}")