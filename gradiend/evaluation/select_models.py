import json
import os

import numpy as np
import torch
from transformers import AutoModel, AutoModelForMaskedLM

from gradiend.evaluation.analyze_decoder import default_evaluation
from gradiend.evaluation.analyze_encoder import get_file_name, analyze_models, get_model_metrics
from gradiend.model import ModelWithGradiend, GradiendModel, AutoModelForLM
from gradiend.util import convert_tuple_keys_to_strings

py_print = print

def select(model, max_size=None, print=True, force=False, plot=True, accuracy_function=lambda x: x, output_suffix="", output=True):
    model_base_name = os.path.basename(model)
    output_result = f'results/models/evaluation_{model_base_name}.json'
    ae = None
    metric_keys = ['bpi', 'fpi', 'mpi']

    if force or accuracy_function is not None or not os.path.isfile(output_result):
        split = 'test'
        file_format = 'csv'
        enc_output = get_file_name(model, max_size=max_size, file_format=file_format, split=split)
        if not os.path.isfile(enc_output):
            py_print(f'Analyze model {model} since file {enc_output} does not exist')
            analysis = analyze_models(model, max_size=max_size, split=split, force=force)
        encoder_metrics = get_model_metrics(enc_output)

        decoder_metrics = default_evaluation(model, large=True, plot=plot, accuracy_function=accuracy_function)

        ae = ModelWithGradiend.from_pretrained(model)

        result = {
            'encoder': encoder_metrics,
            'decoder': decoder_metrics,
            'training': ae.gradiend.kwargs['training']
        }

        # create biased models
        base_model_output = f'results/changed_models/{model_base_name}'
        if force or accuracy_function is not None or not os.path.isdir(base_model_output):
            ae.base_model.save_pretrained(base_model_output)
            ae.tokenizer.save_pretrained(base_model_output)

            version_map = {
                'bpi': 'N',
                'fpi': 'F',
                'mpi': 'M'
            }

            for key in metric_keys:
                key_metrics = decoder_metrics[key]
                lr = key_metrics['lr']
                gender_factor = key_metrics['gender_factor']

                changed_model = ae.modify_model(lr=lr, gender_factor=gender_factor)
                version = version_map[key]
                key_output = f'{base_model_output}-{version}{output_suffix}'
                changed_model.save_pretrained(key_output)
                ae.tokenizer.save_pretrained(key_output)
                py_print(f'Saved {key} model to {key_output} with gender factor {gender_factor} and learning rate {lr}')

                del changed_model
                # release memory
                torch.cuda.empty_cache()


        json_compatible_result = convert_tuple_keys_to_strings(result)
        with open(output_result, 'w') as f:
            json.dump(json_compatible_result, f, indent=2)
    else:
        with open(output_result, 'r') as f:
            result = json.load(f)
            encoder_metrics = result['encoder']
            decoder_metrics = result['decoder']

    if output:
        if ae is None:
            ae = ModelWithGradiend.from_pretrained(model)

        # save the best models to output
        for key in metric_keys:
            key_metrics = decoder_metrics[key]
            lr = key_metrics['lr']
            gender_factor = key_metrics['gender_factor']
            changed_model = ae.modify_model(lr=lr, gender_factor=gender_factor)

            output_path = f'results/changed_models/{model_base_name}-{version_map[key]}{output_suffix}'
            if not os.path.isdir(output_path):
                changed_model.save_pretrained(output_path)
                ae.tokenizer.save_pretrained(output_path)
                py_print(f'Saved {key} model to {output_path}')
            else:
                # check if saved model is the same
                saved_model = AutoModelForLM.from_pretrained(output_path)
                # todo adjust for generative models!
                try:
                    if not np.allclose(saved_model.base_model.embeddings.position_embeddings.weight.cpu().detach(), changed_model.base_model.embeddings.position_embeddings.weight.cpu().detach()):
                        py_print(f'Error: Existing Model {output_path} was not the same as the current model')
                        changed_model.save_pretrained(output_path)
                except AttributeError:
                    py_print(f'WARNING: Model {output_path} does not have position embeddings, skipping check')

    if print:
        py_print(f'Evaluation for model {model}')
        py_print('Encoder:')
        py_print('\tAccuracy Total:', encoder_metrics['acc_total'])
        py_print('\tCorrelation:', encoder_metrics['pearson_total'])
        py_print('\tAccuracy:', encoder_metrics['acc'])
        py_print('\tCorrelation MF:', encoder_metrics['pearson_MF'])
        py_print('\tMA', encoder_metrics['encoded_abs_means'])
        py_print('Decoder:')
        py_print('\tBPI:', decoder_metrics['bpi'])
        py_print('\tFPI:', decoder_metrics['fpi'])
        py_print('\tMPI:', decoder_metrics['mpi'])
        py_print('\tBase model BPI', decoder_metrics['base']['bpi'])
        py_print('\tBase model FPI', decoder_metrics['base']['fpi'])
        py_print('\tBase model MPI', decoder_metrics['base']['mpi'])

    return result


if __name__ == '__main__':
    pass