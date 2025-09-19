import itertools
import os
import time
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from gradiend.util import hash_it


factual_computation_required_keywords = {'factual', 'diff'}
counterfactual_computation_required_keywords = {'counterfactual', 'diff'}
source_target_keywords = {None} | factual_computation_required_keywords | counterfactual_computation_required_keywords

class ModelFeatureTrainingDataset:

    def __init__(self,
                 training_data,
                 tokenizer,
                 feature_creator, # function that creates features from the training data, e.g., model_with_gradiend.forward_pass_create_gradients to create gradients from the training data
                 feature_creator_id, # id of the feature creator, used for caching
                 source='factual',
                 target='diff',
                 cache_dir=None,
                 use_cached_gradients=True,
                 dtype=torch.float32,
                 device=None,
                 return_metadata=False
                 ):
        super().__init__()
        
        assert source in source_target_keywords, f'Invalid source {source}, must be one of {source_target_keywords}'
        assert target in source_target_keywords, f'Invalid target {target}, must be one of {source_target_keywords}'

        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.training_data = training_data
        self.tokenizer = tokenizer
        self.batch_size = self.training_data.batch_size if self.training_data.batch_size else 1
        self.feature_creator = feature_creator
        self.source = source
        self.target = target
        self.cache_dir = cache_dir + f'/{feature_creator_id}' if cache_dir else None
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.use_cached_gradients = use_cached_gradients # if True, gradients will be cached if the cache_dir is set
        self.dtype = dtype
        self.device = device
        self.return_metadata = return_metadata
        self.feature_creator_id = feature_creator_id

    def __len__(self):
        return len(self.training_data) // self.batch_size

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    def __getitem__(self, index):
        indices = list(range(index * self.batch_size, (index + 1) * self.batch_size))

        if len(indices) == 1:
            batch = self.training_data[indices[0]]
        else:
            batch = {}
            for index in indices:
                data = self.training_data[index]
                for key in data:
                    if key not in batch:
                        batch[key] = []
                    batch[key].append(data[key])

            # convert lists of dicts with tensors to dict of tensors
            for key in batch:
                if isinstance(batch[key], list) and all(isinstance(d, dict) for d in batch[key]):
                    first = batch[key][0]
                    first_key = next(iter(first))
                    if not hasattr(first[first_key], 'shape'):
                        raise NotImplementedError('TODO')
                    else:
                        needs_padding = any(
                            any(d[subkey].shape != first[subkey].shape for d in batch[key])
                            for subkey in first
                        )

                    if needs_padding:
                        padded = {}
                        for subkey in first:
                            tensors = [d[subkey] for d in batch[key]]
                            padding_value = self.tokenizer.pad_token_id if 'input_ids' in subkey else 0
                            padded[subkey] = pad_sequence(tensors, batch_first=True, padding_value=padding_value)
                        batch[key] = padded
                    else:
                        batch[key] = {subkey: torch.stack([d[subkey] for d in batch[key]]) for subkey in first}

        cache_file_factual = ''
        cache_file_counterfactual = ''
        if self.use_cached_gradients:
            hash = hash_it([batch['text'], self.dtype])
            cache_file_factual = f'{self.cache_dir}/factual_{hash}.pt'
            cache_file_counterfactual = f'{self.cache_dir}/counterfactual_{hash}.pt'


        factual_gradients = None
        counterfactual_gradients = None

        if self.use_cached_gradients:
            if os.path.exists(cache_file_factual):
                factual_gradients = torch.load(cache_file_factual, weights_only=True)

            if os.path.exists(cache_file_counterfactual):
                counterfactual_gradients = torch.load(cache_file_counterfactual, weights_only=True)

        requires_factual = self.source in factual_computation_required_keywords or self.target in factual_computation_required_keywords
        if factual_gradients is None and requires_factual:
            factual_inputs = batch[True]
            factual_gradients = self.feature_creator(factual_inputs)
            del factual_inputs
            factual_gradients = factual_gradients.to(dtype=self.dtype, device=self.device)

            # save the factual gradients to cache if caching is enabled
            if self.use_cached_gradients and self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(factual_gradients, cache_file_factual)

        requires_counterfactual = self.source in counterfactual_computation_required_keywords or self.target in counterfactual_computation_required_keywords
        if counterfactual_gradients is None and requires_counterfactual:
            counterfactual_inputs = batch[False]
            counterfactual_gradients = self.feature_creator(counterfactual_inputs)
            del counterfactual_inputs

            counterfactual_gradients = counterfactual_gradients.to(dtype=self.dtype, device=self.device)

            # save the counterfactual gradients to cache if caching is enabled
            if self.use_cached_gradients and self.cache_dir is not None:
                os.makedirs(self.cache_dir, exist_ok=True)
                torch.save(counterfactual_gradients, cache_file_counterfactual)

        # determine the source and target tensors based on the source and target keywords in a memory-efficient way
        if self.source == 'factual':
            source_tensor = factual_gradients
        elif self.source == 'diff':
            source_tensor = factual_gradients - counterfactual_gradients
        elif self.source == 'counterfactual':
            source_tensor = counterfactual_gradients
        elif self.source is None:
            source_tensor = None
        else:
            raise ValueError(f'Unknown source: {self.source}')

        target_tensor = factual_gradients
        if self.target == 'counterfactual':
            target_tensor = counterfactual_gradients
        elif self.target == 'diff':
            target_tensor -= counterfactual_gradients
        elif self.target == 'factual':
            pass  # target_tensor is already set
        elif self.target is None:
            target_tensor = None
        else:
            raise ValueError(f'Unknown target: {self.target}')

        del factual_gradients
        del counterfactual_gradients

        output = {
            'source': source_tensor,
            'target': target_tensor,
        }

        for key in batch:
            if key not in output and key != 'metadata':
                output[key] = batch[key]

        if self.return_metadata and 'metadata' in batch:
            output['metadata'] = batch['metadata']

        return output


def create_eval_dataset_v1(gradiend, max_size=None, split='val', source='factual', save_layer_files=False, is_generative=False):
    if not source in {'factual', 'counterfactual', 'diff'}:
        raise ValueError(f'Invalid source {source}')



    start = time.time()
    dataset = create_training_dataset(gradiend.tokenizer, split=split)
    names = dataset.names
    names = names[names['gender'] != 'B']
    texts = dataset.data['masked'].values
    if max_size:
        if 0.0 <= max_size < 1.0:
            max_size = int(max_size * len(texts))
        texts = texts[:max_size]
    mask_token = gradiend.tokenizer.mask_token

    female_names = itertools.cycle(names[names['gender'] == 'F'].iterrows())
    male_names = itertools.cycle(names[names['gender'] == 'M'].iterrows())

    filled_texts = {}
    for text in texts:
        for _, name in [next(female_names), next(male_names)]:

            filled_text = text.replace('[NAME]', name['name'])
            if is_generative:
                filled_text = filled_text.split('[PRONOUN]')[0]
            else:
                filled_text = filled_text.replace('[PRONOUN]', mask_token)
            filled_texts[filled_text] = name['gender']

    # calculate the gradients in advance, if not already cached?
    base_model = gradiend.base_model.name_or_path
    base_model = os.path.basename(base_model)
    layers_hash = gradiend.layers_hash
    cache_dir = f'results/cache/gradients/{base_model}/{source}/{layers_hash}'
    gradients = defaultdict(dict) # maps texts to the gradients
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    is_instruction_model = gradiend.is_instruction_model

    text_iterator = tqdm(filled_texts, desc=f'Loading cached evaluation data', leave=False)
    for i, filled_text in enumerate(text_iterator):
        text_hash = hash_it(filled_text)
        def create_layer_file(layer):
            return f'{cache_dir}/{text_hash}/{layer}.pt'

        cached_tensor_file = f'{cache_dir}/tensor_{text_hash}.pt'
        if os.path.exists(cached_tensor_file):
            gradient = torch.load(cached_tensor_file).half().cpu()
            gradients[filled_text] = gradient
            continue

        # first check whether we need to calculate the gradients
        requires_grad = any(not os.path.exists(create_layer_file(layer)) for layer in gradiend.gradiend.layers)

        # only compute the gradients (computationally expensive) if really needed
        if requires_grad:
            print(f'Calculate gradients for {filled_text}')
            label = filled_texts[filled_text]
            if source == 'diff':
                label_factual = 'he' if label == 'M' else 'she'
                label_counter_factual = 'she' if label == 'M' else 'he'
                if is_instruction_model:
                    label_factual = f' {label_factual}'
                    label_counter_factual = f' {label_counter_factual}'
                inputs_factual = gradiend.create_inputs(filled_text, label_factual)
                grads_factual = gradiend.forward_pass_create_gradients(inputs_factual, return_dict=True)
                inputs_counter_factual = gradiend.create_inputs(filled_text, label_counter_factual)
                grads_counter_factual = gradiend.forward_pass_create_gradients(inputs_counter_factual, return_dict=True)
                grads = {layer: grads_factual[layer] - grads_counter_factual[layer] for layer in gradiend.gradiend.layers}
            else:
                if source == 'factual':
                    label = 'he' if label == 'M' else 'she'
                elif source == 'counterfactual':
                    label = 'she' if label == 'M' else 'he'
                inputs = gradiend.create_inputs(filled_text, label)
                grads = gradiend.forward_pass_create_gradients(inputs, return_dict=True)

            if save_layer_files:
                # create the directory
                dummy_file = create_layer_file('dummy')
                os.makedirs(os.path.dirname(dummy_file), exist_ok=True)
        else:
            grads = None

        for layer in gradiend.gradiend.layers:
            layer_file = create_layer_file(layer)
            if not os.path.exists(layer_file):
                weights = grads[layer].half().flatten().cpu()
                if save_layer_files:
                    # Saving individual layer files doubles the necessary storage, but is more efficient when working with different layer subsets
                    torch.save(weights, layer_file)
            else:
                weights = torch.load(layer_file, weights_only=False)

            weights = weights.float() # convert back to float32 for consistency with other parameters
            gradients[filled_text][layer] = weights

        # convert layer dict to single tensor
        full_gradient = torch.concat([v for v in gradients[filled_text].values()], dim=0).half()
        gradient = full_gradient
        if isinstance(gradiend.gradiend.layers, dict):
            mask = torch.concat(
                [gradiend.gradiend.layers[k].flatten() for k in gradients[filled_text].keys()], dim=0
            ).cpu()
            gradient = full_gradient[mask]

        gradients[filled_text] = gradient
        torch.save(gradient, cached_tensor_file)

    labels = {k: int(v == 'M') for k, v in filled_texts.items()}

    result = {
        'gradients': gradients,
        'labels': labels
    }

    print(f'Loaded the evaluation data with {len(gradients)} entries in {time.time() - start:.2f}s')

    return result
