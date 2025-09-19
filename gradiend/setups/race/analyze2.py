import pandas as pd

from gradiend.model import ModelWithGradiend
from gradiend.setups.race.training import WhiteBlackSetup, WhiteAsianSetup, BlackAsianSetup, ChristianJewishSetup, \
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
    model = 'results/models/race_white_black/distilbert-base-cased-v5'
    model = 'results/models/race_white_asian/bert-base-cased-v2'

    base_models = ['distilbert-base-cased', 'bert-base-cased', 'gpt2', 'bert-large-cased', 'roberta-large', 'Llama-3.2-3B', 'Llama-3.2-3B-Instruct']
    #base_models = ['roberta-large']
    #base_models = ['Llama-3.2-3B', 'Llama-3.2-3B-Instruct']
    setups = [WhiteBlackSetup(), WhiteAsianSetup(), BlackAsianSetup(), ChristianJewishSetup(), ChristianMuslimSetup(), MuslimJewishSetup()]


    #model = 'results/models/race_white_black/gpt2-v5'
    #analyze(setup, model)
    for _ in range(2):
        for base_model in base_models:
            for setup in setups:
                try:
                    print(f'Analyzing {setup.id} with base model {base_model}')
                    model = f'results/models/{setup.id}/{base_model}-v5'
                    analyze(setup, model)

                    #setup.get_model_metrics(f'{model}/encoded_values.csv', trained_state=setup.trained_state)
                except Exception as e:
                    print(f"Error analyzing {model}: {e}")