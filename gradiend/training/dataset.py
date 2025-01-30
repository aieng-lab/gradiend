import itertools
import os
import time
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from gradiend.data import read_processed_bookscorpus, read_geneutral, get_gender_words, read_namexact
from gradiend.util import hash_it



class TrainingDataset(Dataset):

    """Initializes the GRADIEND training dataset.

    Args:
    - data (pd.DataFrame): The GENTER template training data.
    - names (pd.DataFrame): The names dataset.
    - tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for tokenization.
    - batch_size (int): The batch size used for the training. Each data entry will be paired with batch_size random names.
    - neutral_data (pd.DataFrame): The neutral data used for the training between the gender-related maskings. If None, no neutral data is used (default).
     - neutral_data_prop (float): The proportion of neutral data used in the training. If 0, no neutral data is used (default).
     - gender_words (list): A list of gender specific words that are excluded for masking of neutral_data.
    - max_token_length (int): The maximum token length of the input (48 are sufficient for most of the GENTER data)

    In each batch, a random gender is chosen and only this names of this gender are used for the entire batch (with
    different random texts). At the end of the training, each entry of data will be paired exactly batch_size many times with a name.
    This behavior is made deterministic and is automatically applied when using __getitem__.
     """
    def __init__(self, data, names, tokenizer, batch_size=None, neutral_data=None, neutral_data_prop=0.0, gender_words=None, max_token_length=48):
        self.data = data
        self.names = names
        self.names = self.names[self.names['gender'].isin(['M', 'F'])]
        self.tokenizer = tokenizer
        self.mask_token_id = self.tokenizer.mask_token_id
        self.mask_token = self.tokenizer.mask_token
        self.batch_size = None if not batch_size else batch_size
        self.neutral_data = neutral_data if neutral_data_prop > 0 else None
        self.neutral_data_prop = neutral_data_prop
        self.gender_pronoun_tokens = {(gender, upper): self.tokenizer.encode(pronoun[0].upper() + pronoun[1:] if upper else pronoun, add_special_tokens=False)[0] for gender, pronoun in [('M', 'he'), ('F', 'she')] for upper in [True, False]}
        self._analyze_name_tokens()

        gender_words = gender_words or []
        self.gender_token_ids = list(set(token for word in gender_words for token in self.tokenizer(word, add_special_tokens=False)['input_ids']))

        deterministic_random = random.Random(42)
        self.index_sampling = deterministic_random.sample(range(self.len_without_batch()), self.len_without_batch())
        self.full_index_sampling = deterministic_random.sample(range(self.__len__()), len(self))
        self.max_token_length = max_token_length

    """Returns the number of raw entries in the dataset. If the neutral data is used,
    the length is calculated as if the neutral data was part of the main data."""
    def len_without_batch(self):
        n = len(self.data)
        if self.neutral_data is not None:
            n = int(n / (1 - self.neutral_data_prop))
        return n

    """Returns the number of entries in the dataset, considering the batch size."""
    def __len__(self):
        n = self.len_without_batch()

        if self.batch_size == None:
            return n
        else:
            return n * self.batch_size

    def get_data(self, index):
        is_gender_data = True
        data = self.data
        offset = index % self.batch_size if self.batch_size is not None else 0
        idx = index // self.batch_size if self.batch_size is not None else index
        random_idx = self.index_sampling[idx]

        if self.neutral_data is not None and random_idx >= len(self.data):
            is_gender_data = False
            data = self.neutral_data
            random_index = random_idx - len(self.data)
        else:
            random_index = self.full_index_sampling[random_idx]

        final_index = (random_index + offset) % len(data)

        entry = data.iloc[final_index]
        return entry, is_gender_data, idx, offset


    def _analyze_name_tokens(self):
        self.gender_name_tokens = {}

        for gender, df in self.names.groupby('gender'):
            # Dictionary to store the mapping
            token_dict = defaultdict(dict)
            names = df['name'].unique()

            # Tokenize each name and store in the dictionary
            for name in names:
                # Tokenize the name
                tokens = self.tokenizer.encode(name, add_special_tokens=False)
                token_count = len(tokens)

                # Add the name to the dictionary based on the number of tokens
                token_dict[token_count][name] = tokens
            self.gender_name_tokens[gender] = token_dict

        # sanitize, i.e. make sure that the token counts are available for each gender
        # Find common token counts across all genders
        common_token_counts = set(self.gender_name_tokens[next(iter(self.gender_name_tokens))].keys())
        for gender in self.gender_name_tokens:
            common_token_counts.intersection_update(self.gender_name_tokens[gender].keys())

        # Filter the token dictionaries to only include common token counts
        for gender in self.gender_name_tokens:
            self.gender_name_tokens[gender] = {token_count: names
                                          for token_count, names in self.gender_name_tokens[gender].items()
                                          if token_count in common_token_counts}

        self.token_count_distribution = {count: min(len(self.gender_name_tokens[gender][count]) for gender in self.gender_name_tokens)
                                         for count in self.gender_name_tokens['M']}

    def get_random_name(self, gender, seed):
        names = [(k, v) for names in self.gender_name_tokens[gender].values() for k, v in names.items()]
        random_name = random.Random(seed).choice(names)
        return {'name': random_name[0], 'tokens': random_name[1]}

    def tokenize(self, text):
        item = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_token_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in item.items()}
        return item

    def mask_tokens(self, inputs, mask_probability=0.15):
        """
        Mask tokens in the input tensor for MLM, excluding certain tokens.

        Args:
        - inputs (torch.Tensor): The tokenized inputs.
        - mask_probability (float): The probability of masking a token.

        Returns:
        - masked_inputs (torch.Tensor): The inputs with some tokens masked.
        - labels (torch.Tensor): The labels for the MLM task.
        """
        labels = inputs.clone()

        # Create a mask to exclude certain token IDs from masking
        exclude_mask = (inputs.unsqueeze(-1) == torch.Tensor(self.gender_token_ids)).any(dim=-1)

        # Create a random mask for the inputs with the given probability
        random_mask = torch.rand(inputs.shape, dtype=torch.float) < mask_probability

        padding_mask = inputs == self.tokenizer.pad_token_id

        # Apply the exclusion mask to ensure excluded tokens are not masked
        mask = random_mask & ~exclude_mask & ~padding_mask

        # Mask the tokens in the inputs
        masked_inputs = inputs.clone()
        masked_inputs[mask] = self.mask_token_id

        # For labels, we only want to compute loss on masked tokens
        labels[~mask] = -100

        return masked_inputs, labels


    def __getitem__(self, index):
        entry, is_gender_data, batch_size_index, batch_size_offset = self.get_data(index)

        if is_gender_data:
            masked = entry['masked']

            gender = random.Random(batch_size_index).choice(['M', 'F'])
            inverse_gender = 'F' if gender == 'M' else 'M'

            random_name = self.get_random_name(gender, batch_size_offset)

            def fill(text, name):
                return text.replace('[NAME]', name).replace('[PRONOUN]', self.mask_token)
            gender_text = fill(masked, random_name['name'])
            #inv_gender_text = fill(masked, inv_gender_name)

            item = self.tokenize(gender_text)
            gender_labels = item['input_ids'].clone()
            mask_token_mask = gender_labels == self.mask_token_id
            gender_labels[~mask_token_mask] = -100 # only compute loss on masked tokens

            inv_gender_labels = gender_labels.clone()

            # check if the pronoun start with an uppercase letter since it appears at the beginning of the sentence
            gender_text_no_white_spaces = gender_text.replace(' ', '')
            mask_index = gender_text_no_white_spaces.index(self.mask_token)
            upper = mask_index == 0 or mask_index > 2 and gender_text_no_white_spaces[mask_index - 1] in {'.', '!', '?'} and gender_text_no_white_spaces[mask_index - 2] != '.' # avoid "..."
            gender_labels[mask_token_mask] = self.gender_pronoun_tokens[(gender, upper)] # set masked tokens to the
            inv_gender_labels[mask_token_mask] = self.gender_pronoun_tokens[(inverse_gender, upper)] # set masked tokens to the

            inv_item = item.copy()
            item['labels'] = gender_labels
            inv_item['labels'] = inv_gender_labels

            label = int(gender == 'M') # 1 -> M, 0 -> F
            text = gender_text
        else:
            # this dataset is used for general knowledge data (not gender-specific)
            text = entry['text']
            item = self.tokenize(text)
            masked_input, labels = self.mask_tokens(item['input_ids'])
            item['input_ids'] = masked_input
            item['labels'] = labels
            inv_item = item
            label = ''

        return {True: item, False: inv_item, 'text': text, 'label': label}



def create_name_dataset(tokenizer, max_size=None, batch_size=None, neutral_data=False, split=None, neutral_data_prop=0.5):
    # Load dataset
    names = read_namexact(split=split)
    dataset = read_processed_bookscorpus(split=split)
    if max_size:
        dataset = dataset.iloc[range(max_size)]

    neutral_data = read_geneutral(max_size=len(dataset), split=split) if neutral_data else None

    gender_words = get_gender_words()

    # Create custom dataset
    return TrainingDataset(dataset, names, tokenizer, batch_size=batch_size, neutral_data=neutral_data, gender_words=gender_words, neutral_data_prop=neutral_data_prop)


def create_eval_dataset(bert_with_ae, max_size=None, split='val', source='gradient', save_layer_files=False):
    if not source in {'gradient', 'inv_gradient', 'diff'}:
        raise ValueError(f'Invalid source {source}')

    start = time.time()
    dataset = create_name_dataset(bert_with_ae.tokenizer, split=split)
    names = dataset.names
    names = names[names['gender'] != 'B']
    texts = dataset.data['masked'].values
    if max_size:
        if 0.0 <= max_size < 1.0:
            max_size = int(max_size * len(texts))
        texts = texts[:max_size]
    mask_token = bert_with_ae.tokenizer.mask_token

    female_names = itertools.cycle(names[names['gender'] == 'F'].iterrows())
    male_names = itertools.cycle(names[names['gender'] == 'M'].iterrows())

    filled_texts = {}
    for text in texts:
        for _, name in [next(female_names), next(male_names)]:
            filled_text = text.replace('[NAME]', name['name']).replace('[PRONOUN]', mask_token)
            filled_texts[filled_text] = name['gender']

    # calculate the gradients in advance, if not already cached?
    base_model = bert_with_ae.bert.name_or_path
    base_model = os.path.basename(base_model)
    cache_dir = f'data/cache/gradients/{base_model}/{source}'
    gradients = defaultdict(dict) # maps texts to the gradients
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    text_iterator = tqdm(filled_texts, desc=f'Loading cached evaluation data', leave=False)
    layers_hash = hash_it(bert_with_ae.ae.layers)
    for i, filled_text in enumerate(text_iterator):
        text_hash = hash_it(filled_text)
        def create_layer_file(layer):
            return f'{cache_dir}/{text_hash}/{layer}.pt'

        cached_tensor_file = f'{cache_dir}/tensor_{text_hash}_{layers_hash}.pt'
        if os.path.exists(cached_tensor_file):
            gradient = torch.load(cached_tensor_file)
            gradients[filled_text] = gradient.half().cpu()
            continue

        # first check whether we need to calculate the gradients
        requires_grad = any(not os.path.exists(create_layer_file(layer)) for layer in bert_with_ae.ae.layers)

        # only compute the gradients (computationally expensive) if really needed
        if requires_grad:
            print(f'Calculate gradients for {filled_text}')
            label = filled_texts[filled_text]
            if source == 'diff':
                label_factual = 'he' if label == 'M' else 'she'
                label_counter_factual = 'she' if label == 'M' else 'he'
                inputs_factual = bert_with_ae.create_inputs(filled_text, label_factual)
                grads_factual = bert_with_ae.forward_pass(inputs_factual, return_dict=True)
                inputs_counter_factual = bert_with_ae.create_inputs(filled_text, label_counter_factual)
                grads_counter_factual = bert_with_ae.forward_pass(inputs_counter_factual, return_dict=True)
                grads = {layer: grads_factual[layer] - grads_counter_factual[layer] for layer in bert_with_ae.ae.layers}
            else:
                if source == 'gradient':
                    label = 'he' if label == 'M' else 'she'
                elif source == 'inv_gradient':
                    label = 'she' if label == 'M' else 'he'
                inputs = bert_with_ae.create_inputs(filled_text, label)
                grads = bert_with_ae.forward_pass(inputs, return_dict=True)

            if save_layer_files:
                # create the directory
                dummy_file = create_layer_file('dummy')
                os.makedirs(os.path.dirname(dummy_file), exist_ok=True)
        else:
            grads = None

        for layer in bert_with_ae.ae.layers:
            layer_file = create_layer_file(layer)
            if not os.path.exists(layer_file):
                weights = grads[layer].half().flatten()
                if save_layer_files:
                    # Saving individual layer files doubles the necessary storage, but is more efficient when working with different layer subsets
                    torch.save(weights, layer_file)
            else:
                weights = torch.load(layer_file, weights_only=False)

            weights = weights.float() # convert back to float32 for consistency with other parameters
            gradients[filled_text][layer] = weights

        # convert layer dict to single tensor
        gradient = torch.concat([v for v in gradients[filled_text].values()], dim=0).half().cpu()
        gradients[filled_text] = gradient
        torch.save(gradient, cached_tensor_file)

    labels = {k: int(v == 'M') for k, v in filled_texts.items()}

    result = {
        'gradients': gradients,
        'labels': labels
    }

    print(f'Loaded the evaluation data with {len(gradients)} entries in {time.time() - start:.2f}s')

    return result

