import itertools
import json
import os
import random
import time

import numpy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from tqdm import tqdm
from skopt import gp_minimize, dump, load
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
import matplotlib.pyplot as plt

from gradiend.data import read_gerneutral
from gradiend.data.util import get_file_name, json_loads, json_dumps
from gradiend.evaluation.analyze_decoder import get_evaluation_file, convert_results_to_dict, convert_results_to_list, \
    compute_bias_score, compute_lms, plot_gradiend_model_selection
from gradiend.setups import BatchedTrainingDataset, Setup
from gradiend.setups.gender.en import read_geneutral
from gradiend.setups.race.data import _create_bias_attribute_words, get_base_terms
from gradiend.training.decoder_only_mlm.model import DecoderModelWithMLMHead
from gradiend.training.gradiend_training import train_for_configs
from gradiend.util import hash_model_weights, get_files_and_folders_with_prefix
from gradiend.setups.race.util import class2pretty_name

class TrainingDataset(BatchedTrainingDataset):
    def __init__(self,
                 data,
                 tokenizer,
                 batch_size=None,
                 is_generative=False,
                 max_size=None,
                 target_key='label',
                 ):
        super().__init__(data=data,
                         tokenizer=tokenizer,
                         max_size=max_size,
                         batch_size=batch_size,
                         batch_criterion=target_key,
                         balance_column='df_ctr',
                         )
        self.target_key = target_key
        if is_generative:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.mask_token = self.tokenizer.mask_token
        self.is_generative = is_generative

        self.is_instruct_model = 'instruct' in self.tokenizer.name_or_path.lower()

    def __getitem__(self, idx):
        entry = super().__getitem__(idx) # Get the adjusted index for batching

        if self.is_generative:
            raw_text = entry['masked'].split('[MASK]')[0].strip()
        else:
            raw_text = entry['masked'].replace('[MASK]', self.mask_token)

        label = entry[self.target_key]
        if isinstance(self.target_key, list):
            label = tuple(int(l) for l in label)
        elif isinstance(label, numpy.int_):
            label = int(label)

        item_factual = self._create_item(raw_text, entry['source'])
        item_counterfactual = self._create_item(raw_text, entry['target'])

        return {
            True: item_factual,
            False: item_counterfactual,
            'text': raw_text,
            'full_text': entry['masked'],
            'label': label,
            'binary_label': label,
            'factual_target': entry['source'],
            'counterfactual_target': entry['target'],
            'source': entry['source'],
            'source_id': entry['source_id'],
            'target': entry['target'],
            'target_id': entry['target_id'],

            #'metadata': {},

        }


def create_training_dataset(tokenizer,
                            max_size=None,
                            batch_size=None,
                            neutral_data=False,
                            split='train',
                            neutral_data_prop=0.5,
                            races=None,
                            bias_type='race',
                            single_word_texts=False,
                            is_generative=None,
                            ):
    if races is None or len(races) < 2:
        raise ValueError("Please provide at least two races in the 'races' list.")

    all_dfs = []
    label_counter = 0
    is_generative = is_generative or tokenizer.mask_token_id is None

    vocab = set(tokenizer.get_vocab().keys())

    pair_to_index = {
        frozenset((i, j)): idx
        for idx, (i, j) in enumerate(itertools.combinations(range(len(races)), 2))
    }
    num_pairs = len(pair_to_index)
    df_ctr = 0

    # Load all pairwise datasets
    for i, race_from in enumerate(races):
        for j, race_to in enumerate(races):
            if i == j:
                continue  # skip same-race
            ds_path = f"data/{bias_type}/{race_from}_to_{race_to}"
            try:
                df = Dataset.load_from_disk(ds_path).to_pandas()
            except FileNotFoundError:
                continue  # skip if dataset doesn't exist

            pair_idx = pair_to_index[frozenset((i, j))]

            # add index label
            if num_pairs == 1:
                df["label_idx"] = label_counter
                label_counter += 1
            else:
                df["label_idx"] = pair_idx

            # add one-hot columns
            one_hot = np.zeros(num_pairs, dtype=int)
            one_hot[pair_idx] = 1 if i > j else -1
            for k in range(num_pairs):
                df[f"label_{k}"] = one_hot[k]

            # Deterministic split
            if split:
                # todo save the splits persistently
                train, val, test = 0.7, 0.2, 0.1
                df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                n_train = int(len(df) * train)
                n_val = int(len(df) * val)
                if split == "train":
                    df = df[:n_train]
                elif split == "val":
                    df = df[n_train:n_train + n_val]
                elif split == "test":
                    df = df[n_train + n_val:]
                else:
                    raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")

            is_llama = 'llama' in tokenizer.name_or_path.lower()
            if not is_llama:
                df = df[df["source"].isin(vocab) & df["target"].isin(vocab)]

            if df.empty:
                raise ValueError('No source and target words in vocab!')

            df['df_ctr'] = df_ctr
            df_ctr += 1
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No datasets found for the given races.")

    # ensure all datasets have the same number of examples
    min_size = min(len(df) for df in all_dfs)
    all_dfs = [df.sample(min_size, random_state=42).reset_index(drop=True) for df in all_dfs]

    # Merge datasets
    templates = pd.concat(all_dfs, ignore_index=True).sample(frac=1.0).reset_index(drop=True)

    if max_size:
        # Stratified sample across all labels
        templates = templates.groupby('label_idx').apply(
            lambda x: x.sample(min(len(x), max_size // templates['label_idx'].nunique()))
        ).reset_index(drop=True)

    if num_pairs == 1:
        target_keys = 'label_idx'
    else:
        target_keys = [f'label_{i}' for i in range(num_pairs)]

    return TrainingDataset(
        templates,
        tokenizer,
        batch_size=batch_size,
        is_generative=is_generative,
        target_key=target_keys,
    )


class BiasSetup(Setup):
    def __init__(self, bias_type, *races, pretty_id=None):
        super().__init__(f'{bias_type}_{"_".join(races)}')
        assert len(races) > 1
        self.races = races
        self.bias_type = bias_type
        self.metric_keys = [f'bias_{self.bias_type}']
        self.metric_keys = [self.id]
        self.data_cache = {}

        bias_attribute_words = _create_bias_attribute_words(
            f"data/bias_attribute_words.json",
            bias_type=self.bias_type,
        )
        base_terms = get_base_terms(bias_attribute_words)
        # todo check that all races are contained in base_terms
        race_base_terms = {t: index for index, t in enumerate(base_terms) if t in self.races}
        self.all_races = base_terms
        self.targets = {t: [tt[i] for tt in bias_attribute_words] for t, i in race_base_terms.items()}
        self.non_neutral_terms = [tt for t in bias_attribute_words for tt in t]

        self.races2labelidx = {}
        self.races2labelidx[(base_terms[0], base_terms[1])] = 0
        self.races2labelidx[(base_terms[0], base_terms[2])] = 1
        self.races2labelidx[(base_terms[1], base_terms[2])] = 2

        self._pretty_id = pretty_id or self.id

    @property
    def pretty_id(self):
        return self._pretty_id

    @property
    def trained_state(self):
        return '<->'.join(sorted(self.races))

    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(*args, races=self.races, bias_type=self.bias_type, **kwargs)

    def analyze_models(self, *models, max_size=1000, force=False, split='test', prefix=None, best_score=None):
        if prefix:
            # find all models in the folder with the suffix
            best_score = '_best' if best_score else ''
            models = list(models) + get_files_and_folders_with_prefix(prefix, only_folder=True, suffix=best_score)
        print(f'Analyze {len(models)} Models:', models)


        dfs = {}
        for model in models:

            output = get_file_name(model, max_size=max_size, file_format='csv', split=split)

            analyze_df = None
            if False and force or not os.path.isfile(output):
                from gradiend.model import ModelWithGradiend
                model_with_gradiend = ModelWithGradiend.from_pretrained(model)
                # todo
                #analyze_df = self.analyze_model(model_with_gradiend, output=output, split=split)
                print(f'Done with Model {model}')

            else:
                print(f'Skipping Model {model} as output file {output} already exists!')
                analyze_df = pd.read_csv(output)

            if len(models) == 1:
                return analyze_df
            dfs[model] = analyze_df
        return dfs

    def _get_default_prediction_file_name(self, model):
        model_name = model.name_or_path if hasattr(model, 'name_or_path') else model
        return f'results/cache/{self.id}/default_predictions_{model_name}.csv'

    def _read_default_predictions(self, model):
        file = self._get_default_prediction_file_name(model)
        try:
            cache_default_predictions = pd.read_csv(file)
            cache_default_predictions.set_index('text', inplace=True)
            cache_default_predictions['most_likely_token'] = cache_default_predictions['most_likely_token'].apply(json_loads)
            cache_default_predictions['label'] = cache_default_predictions['label'].apply(json_loads)

            for target in self.targets:
                cache_default_predictions[target] = cache_default_predictions[self.targets].apply(json_loads)
            cache_default_predictions_dict = cache_default_predictions.to_dict(orient='index')
        except FileNotFoundError:
            cache_default_predictions_dict = {}
        return cache_default_predictions_dict

    def __evaluate(self, model, tokenizer, masked_text):
        """
        Evaluate the model on masked language modeling (MLM) task. Specifically, determine the probabilities of the tokens
        he and she in the masked text.

        Args:
        - model: The BERT model (BertForMaskedLM).
        - tokenizer: The BERT tokenizer.
        - masked_text: The text with a masked token (e.g., "The capital of France is [MASK].").
        """
        # Tokenize the input text
        inputs = tokenizer(masked_text, return_tensors="pt")
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get the index of the masked token
        is_generative = tokenizer.mask_token_id is None
        if is_generative:
            mask_token_index = len(inputs["input_ids"]) - 1
        else:
            mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

        # Pass the inputs through the model
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract the logits and softmax to get probabilities
        logits = outputs.logits
        if isinstance(model, DecoderModelWithMLMHead):
            mask_token_logits = logits
        else:
            mask_token_logits = logits[0, mask_token_index, :]
        probabilities = torch.softmax(mask_token_logits, dim=-1)

        # Get the token IDs for "he" and "she"
        raw_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        target_token_ids = {s: raw_tokenizer.convert_tokens_to_ids(s) for s in self.targets}

        # Get probabilities for targets
        result = {}
        for t, tid in target_token_ids.items():
            if tid == tokenizer.unk_token_id:  # target not in vocab
                result[t] = None
            else:
                result[t] = probabilities[tid].item()

        # Most likely token overall
        most_likely_token_id = torch.argmax(probabilities).item()
        most_likely_token = tokenizer.decode([most_likely_token_id])

        result["most_likely_token"] = most_likely_token
        return result


    def analyze_model(self,
                      model_with_gradiend,
                      output=None,
                      split='test',
                      plot=False,
                      max_size=10000,
                      force=False,
                      ):


        if not force and os.path.isfile(output):
            try:
                df = pd.read_csv(output)
                mask = df['state'] == 'train-neutral'
                if not mask.any():
                    raise ValueError(f'Output file {output} does not contain train-neutral examples, redoing analysis!')
                print(f'Skipping analysis for model {model_with_gradiend.name_or_path} as output file {output} already exists!')
            except Exception as e:
                print(f'Could not read existing output file {output}, redoing analysis!', e)
                force = True

        if force or output is None or not os.path.isfile(output):
            data = self.read_data(model_with_gradiend.tokenizer, max_size=6 * max_size, split=split, races=self.all_races)
            source = model_with_gradiend.gradiend.kwargs['training']['config']['source']

            dataloader = DataLoader(data, batch_size=1, shuffle=False)
            df = data.data
            like_training_data = df[df['source_id'].isin(self.races) & df['target_id'].isin(self.races)]

            for c in ['label_0', 'label_1', 'label_2']:
                labels = like_training_data[c]
                if labels.min() == -1 and labels.max() == 1:
                    label_column = c
                    break
            else:
                raise ValueError('Could not find label column with -1/1 labels')

            source2label = {}
            for gradiend_source in self.races:
                sub_df = like_training_data[like_training_data['source_id'] == gradiend_source]
                labels = sub_df[label_column].unique()
                if len(labels) > 1:
                    raise ValueError(f'Source {gradiend_source} has multiple labels: {labels}')
                source2label[gradiend_source] = labels[0].item()



            if source == 'factual':
                key = True
            elif source == 'counterfactual':
                key = False
            else:
                raise ValueError(f'Unsupported source {source}')

            data = {key: [] for key in {'text', 'source', 'target', 'source_id', 'target_id', 'encoded', 'input', 'label', 'all_labels', 'state', 'symmetric_state'}}

            for i, batch in enumerate(tqdm(dataloader, desc="Encoding texts")):
                input = batch[key]
                batch_gradients = model_with_gradiend.forward_pass_create_gradients(input)
                batch_encoded = model_with_gradiend.gradiend.encoder(batch_gradients)
                batch_encoded = batch_encoded.tolist()
                if len(batch_encoded) == 1:
                    batch_encoded = batch_encoded[0]

                source_id = batch['source_id'][0]
                target_id = batch['target_id'][0]
                data['encoded'].append(batch_encoded)
                label = batch['label']
                if hasattr(label, 'item'):
                    data['all_labels'].append(label.item())
                    data['label'].append(label.item())
                else:
                    labels = tuple(int(l) for l in label)
                    data['all_labels'].append(labels)

                    # check whether we need to invert the label to match our expectation
                    # todo check factual/counterfactual
                    label = source2label.get(source_id, 0)

                    data['label'].append(label)

                data['input'].append(batch[f'{source}_target'][0])
                data['text'].append(batch['full_text'][0])
                data['state'].append(source_id + '->' + target_id)
                data['symmetric_state'].append('<->'.join(sorted([source_id, target_id])))
                data['source'].append(batch['source'][0])
                data['source_id'].append(source_id)
                data['target'].append(batch['target'][0])
                data['target_id'].append(target_id)
            data['type'] = 'bias'

            df = pd.DataFrame(data)

            ###### Evaluation on GERNEUTRAL ############
            texts = []
            encoded_values = []
            labels = []
            torch.manual_seed(42)
            size_per_type = df.groupby('symmetric_state').size().min()
            gerneutral = read_gerneutral(max_size=size_per_type, exclude=self.non_neutral_terms)
            gerneutral_texts = gerneutral['text'].tolist()

            for text in tqdm(gerneutral_texts, desc='GENEUTRAL data'):
                encoded, masked_text, label = model_with_gradiend.mask_and_encode(text, return_masked_text=True)
                if encoded is None:
                    continue

                texts.append(text)
                encoded_values.append(encoded)
                labels.append(label)

            df2 = pd.DataFrame({
                'text': texts,
                'encoded': encoded_values,
                'label': 0,
                'input': labels,
                'type': 'neutral',
                'state': 'neutral',
                'symmetric_state': 'neutral',
                'source': 'neutral',
                'source_id': 'neutral',
                'target': 'neutral',
                'target_id': 'neutral',
            })


            ######## Evaluation on training data but on neutral tokens ############
            texts = []
            encoded_values = []
            labels = []
            torch.manual_seed(42)
            training_like_texts = like_training_data['text'].tolist()
            ignore_tokens = list(set([tt for t in self.non_neutral_terms for tt in model_with_gradiend.tokenizer(t)['input_ids']]))
            for text in tqdm(training_like_texts, desc='Training data without gender words masked'):
                encoded, masked_text, label = model_with_gradiend.mask_and_encode(text,
                                                                                  ignore_tokens=ignore_tokens,
                                                                                  return_masked_text=True)
                if encoded is None:
                    continue

                texts.append(text)
                encoded_values.append(encoded)
                labels.append(label)

            df3 = pd.DataFrame({
                'text': texts,
                'encoded': encoded_values,
                'label': 0,
                'input': labels,
                'type': 'neutral',
                'state': 'train-neutral',
                'symmetric_state': 'train-neutral',
                'source': 'train-neutral',
                'source_id': 'train-neutral',
                'target': 'train-neutral',
                'target_id': 'train-neutral',
            })


            total_results = pd.concat([df, df2, df3], ignore_index=True)
            if output is not None:
                os.makedirs(os.path.dirname(output), exist_ok=True)
                total_results.to_csv(output, index=False)
                print(f'Saved analysis results to {output}')
        else:
            print(f'Skipping analysis for model {model_with_gradiend.name_or_path} as output file {output} already exists!')
            total_results = pd.read_csv(output)



        if plot:
            plot_output = None
            if output:
                plot_output = output.replace('.csv', '.pdf')
            self.plot_split_violin(total_results, output=plot_output)

        if False:
            self.print_examples_by_modes(total_results)

        return total_results

    def get_model_metrics(self, encoded_values, trained_state=None, plot=True, plot_output=None):
        trained_state = trained_state or '<->'.join(sorted(self.races))

        output_str = None
        if isinstance(encoded_values, str):
            output_str = f'{encoded_values.removesuffix(".csv")}_metrics.json'
            encoded_values = pd.read_csv(encoded_values)

        metrics = {}

        # compute corr by state
        for state, group in encoded_values.groupby('symmetric_state'):
            if 'neutral' in state:
                continue

            labels = group['label'].unique()
            if len(labels) < 2:
                print(f"Skipping correlation for state {state} as it has only one label: {labels}")
                metrics[f'corr_{state}'] = None
                continue

            score = np.corrcoef(group['encoded'], group['label'])[0, 1]
            print(f"Correlation for state {state}: {score:.4f}")
            metrics[f'corr_{state}'] = score

            if state == trained_state and score < -0.5:
                # we need to invert the encoding
                encoded_values['encoded'] = -encoded_values['encoded']
                # save the updated encoded values
                encoded_values.to_csv(output_str.replace('_metrics.json', '.csv'), index=False)
                encoded_values['inverted'] = True
                # raise an exception, s.t. the computation is redone with the inverted encoding
                raise ValueError(f'Inverted encoding for trained state {state} as correlation was {score:.4f}')

            if state == trained_state:
                # compute accuracy with decision boundary 0.0
                predictions = (group['encoded'] > 0).astype(int) * 2 - 1  # convert to -1, 1
                accuracy = (predictions == group['label']).mean()
                print(f"Accuracy for state {state}: {accuracy:.4f}")
                mean_abs_encoded_value = group['encoded'].abs().mean()

                metrics[f'acc_{state}'] = accuracy
                metrics[f'acc_train'] = accuracy
                metrics[f'mean_abs_encoded_value_train'] = mean_abs_encoded_value
                metrics[f'mean_abs_encoded_value_{state}'] = mean_abs_encoded_value

        # mark the correlation of the trained state separately
        metrics[f'corr_train'] = metrics.get(f'corr_{trained_state}', None)

        # compute correlation with trained_state and GENEUTRAL
        if trained_state:
            trained_group = encoded_values[encoded_values['symmetric_state'] == trained_state]
            neutral_group = encoded_values[encoded_values['symmetric_state'] == 'neutral']
            if not trained_group.empty and not neutral_group.empty:
                combined = pd.concat([trained_group, neutral_group])
                score = np.corrcoef(combined['encoded'], combined['label'])[0, 1]
                print(f"Correlation for trained state {trained_state} + GERNEUTRAL: {score:.4f}")
                metrics[f'corr_GERNEUTRAL'] = score

                # compute accuracy with decision boundaries -0.5 and 0.5
                def classify(x):
                    if x < -0.5:
                        return -1
                    elif x > 0.5:
                        return 1
                    else:
                        return 0
                predictions = combined['encoded'].apply(classify)
                accuracy = (predictions == combined['label']).mean()
                print(f"Accuracy for trained state {trained_state} + GERNEUTRAL: {accuracy:.4f}")
                metrics[f'acc_GERNEUTRAL'] = accuracy

            else:
                metrics[f'corr_GERNEUTRAL'] = None
                metrics[f'acc_GERNEUTRAL'] = None

        # Mean Abs Encoded value of GENEUTRAL
        neutral_group = encoded_values[encoded_values['symmetric_state'] == 'neutral']
        if not neutral_group.empty:
            mean_abs_encoded_value = neutral_group['encoded'].abs().mean()
            metrics[f'mean_abs_encoded_value_GENEUTRAL'] = mean_abs_encoded_value
        else:
            metrics[f'mean_abs_encoded_value_GENEUTRAL'] = None

        neutral_group = encoded_values[encoded_values['symmetric_state'] == 'train-neutral']
        if not neutral_group.empty:
            mean_abs_encoded_value = neutral_group['encoded'].abs().mean()
            metrics[f'mean_abs_encoded_value_TRAINEUTRAL'] = mean_abs_encoded_value
        else:
            metrics[f'mean_abs_encoded_value_TRAINeutral'] = None

        # compute full correlation
        score = np.corrcoef(encoded_values['encoded'], encoded_values['label'])[0, 1]
        print(f"Overall Correlation: {score:.4f}")
        metrics['corr'] = score

        if output_str:
            os.makedirs(os.path.dirname(output_str), exist_ok=True)
            json.dump(metrics, open(output_str, 'w'))
            print(f'Saved metrics to {output_str}')

        if plot:
            self.plot_split_violin(encoded_values, output=plot_output)

        return metrics

    def plot_split_violin(self, total_results, output=None):
        import os
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        import re
        df = total_results.copy()

        # helper: parse the 'all_labels' column like in your original code
        def parse_label_value(label, label_idx):
            if isinstance(label, str):
                if label.startswith('(') and label.endswith(')'):
                    label = tuple(int(l) for l in label[1:-1].split(','))
                else:
                    label = int(label)
            if isinstance(label, tuple):
                return label[label_idx]
            else:
                # label is scalar and label_idx should be 0
                if label_idx > 0:
                    raise ValueError(f"Label index {label_idx} out of range for label {label}")
                return label


        # compute canonical (left, right, pair_label, source)
        df.loc[df['type'] == 'neutral', 'state'] = df['state']

        # map states to pretty names
        races2labelidx = self.races2labelidx.copy()
        all_races = self.all_races.copy()
        for x, y in class2pretty_name.items():
            df['state'] = df['state'].str.replace(x, y, regex=False)
            for r1, r2 in list(races2labelidx.keys()):
                if r1 == x:
                    races2labelidx[(y, r2)] = races2labelidx.pop((r1, r2))
                if r2 == x:
                    races2labelidx[(r1, y)] = races2labelidx.pop((r1, r2))
            if x in all_races:
                all_races = [r for r in all_races if r != x] + [y]
        all_races = list(sorted(all_races))
        df['source_id'] = df['source_id'].map(lambda x: class2pretty_name.get(x, x))
        df['target_id'] = df['target_id'].map(lambda x: class2pretty_name.get(x, x))
        important_states = all_races


        pair_labels = []
        sources = []
        for _, row in df.iterrows():
            raw_state = str(row["state"]).strip()
            # neutral handling: both GENEUTRAL and TRAIN-NEUTRAL are substates of "neutral"
            if "neutral" in raw_state.lower():
                pair_label = raw_state
                source = raw_state
                left
            else:
                # directed form "s1->s2"
                if "->" not in raw_state:
                    raise ValueError(f"Unexpected state format (expected 's1->s2' or neutral): {raw_state}")
                s1, s2 = [p.strip() for p in raw_state.split("->", 1)]
                state_pair = (s1, s2)
                # if not present in mapping, swap order (as in your original code)
                if state_pair not in races2labelidx:
                    s1, s2 = s2, s1
                label_idx = races2labelidx[(s1, s2)]
                label_val = parse_label_value(row["all_labels"], label_idx)
                # decide canonical sorted order and arrow direction as in your original code
                states_sorted = (s1, s2)
                # your original comparison: if index(s1) < index(s2) then states_sorted=(s2,s1)
                if all_races.index(s1) < all_races.index(s2):
                    states_sorted = (s2, s1)
                if label_val == 1:
                    left, right = states_sorted[0], states_sorted[1]
                elif label_val == -1:
                    left, right = states_sorted[1], states_sorted[0]
                else:
                    raise ValueError(f"Unexpected label {label_val} for state pair {(s1, s2)}")
                pair_label = f"{left} → {right}"
                source = left  # we group on the left-side of the arrow
            pair_labels.append(pair_label)
            sources.append(source)
        # attach computed columns
        df = df.assign(pair=pair_labels, source=sources)

        # compute direction deterministically: 0 if matches the left substate, 1 if matches the right
        def compute_direction(row):
            raw_state = str(row["state"]).strip()
            if 'neutral' in raw_state.lower():
                return int(raw_state == 'neutral')
            # raw_state looks like s1->s2
            s1 = raw_state.split("->", 1)[0].strip()
            s2 = raw_state.split("->", 1)[1].strip()
            states_without_s1 = [s for s in all_races if s1 != s]
            return 0 if s2 == states_without_s1[0] else 1

        df["direction"] = df.apply(compute_direction, axis=1)
        # Build pair ordering grouped by source (left)
        # Use the order of self.all_races where possible, then neutral last if present
        unique_sources = []
        # collect sources in the order of self.all_races (if present), to keep stable ordering
        for r in all_races:
            if r in df["source_id"].values:
                unique_sources.append(r)
        if "neutral" in df["source_id"].values and "neutral" not in unique_sources:
            unique_sources.append("neutral")
        # For each source, collect pair labels in deterministic order (sorted by pair_right for stability)
        pair_order = []
        source_order = list(sorted([s for s in unique_sources if s != "neutral"])) + ['neutral']

        for src in source_order:
            if src == 'neutral':
                prs = [r'\traindatazero', '\gerneutral']
            else:
                prs = sorted(df.loc[df["source_id"] == src, "pair"].unique())
            pair_order.extend(prs)

        # create categorical to preserve plotting order
        df["pair"] = pd.Categorical(df["pair"], categories=pair_order, ordered=True)

        train_mask = df['source_id'].isin(important_states) & df['target_id'].isin(important_states)
        df['train'] = False
        df.loc[train_mask, 'train'] = True
        df.loc[df['source_id'] == 'train-neutral', 'source_id'] = 'Neutral'

        # Build base color per pair (distinct color per pair)
        n_pairs = max(1, len(pair_order))
        base_palette = sns.color_palette("Paired", n_colors=n_pairs)

        # Plot: x = pair (each directed pair), hue = direction (0/1) so split=True works
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(
            data=df,
            x="source_id",
            y="encoded",
            hue="direction",
            split=True,
            inner="quart",  # or "box" if you prefer
            order=source_order,
            #palette=["lightgray", "darkgray"],  # temporary; we recolor manually below
            density_norm="width",
            linewidth=0.7,
            common_norm=False,
            zorder=3,  # draw violins above grid
        )

        train_map = {}
        for (src, dir_), group in df.groupby(['source_id', 'direction']):
            # All rows in this group should have same train status
            train_map[(src, dir_)] = group['train'].iloc[0]

        # Adjust linewidths per half-violin
        for i, violin in enumerate(ax.collections):
            # In Seaborn split violins, each half-violin is one collection
            # The order is predictable: for each x-tick, the halves are ordered by hue
            x_tick = i // 2  # integer division: which source
            hue_val = i % 2  # 0 or 1 depending on hue order
            src_val = df['source_id'].unique()[x_tick]
            dir_val = sorted(df['direction'].unique())
            if hue_val > len(dir_val) - 1:
                raise ValueError(f"Unexpected hue value {hue_val} for violin {i}")
            dir_val = dir_val[hue_val]  # get actual direction value (0/1)
            # Seaborn orders hues sorted by default
            if (src_val, dir_val) in train_map and train_map[(src_val, dir_val)]:
                violin.set_linewidth(2.5)
            else:
                violin.set_linewidth(0.5)

        # print the label per source_id
        df['pretty_source_id'] = df['source_id'].map(lambda x: class2pretty_name.get(x, x))
        for xtick, _label in enumerate(ax.get_xticklabels()):
            label = df[df['pretty_source_id'].str.lower() == _label.get_text().lower()]['label'].unique()
            if len(label) != 1:
                raise ValueError(f"Unexpected multiple labels for source_id {label} ({_label.get_text()})")
            label = label[0]
            # plot a red dot at the y position of the label
            ax.plot(xtick, label, 'ro', markersize=10, zorder=20, markeredgecolor='black', markerfacecolor='yellow')

        # Build base color palette: enough colors for all halves
        n_halves = len(pair_order) * 2  # left + right per pair
        base_palette = sns.color_palette("Paired", n_colors=n_halves)

        # Recolor each half-violin with distinct color
        collections = ax.collections
        for i, poly in enumerate(collections):
            if i >= n_halves:
                continue  # skip medians or extra polygons
            color = base_palette[i]
            poly.set_facecolor(color)
            poly.set_edgecolor("black")
            poly.set_alpha(1.0)

        # hide original per-pair labels and set grouped labels at midpoints
        ax.set_xticks(range(len(unique_sources)))
        ax.set_xticklabels(source_order, fontsize=12)

        # --- Train status legend (linewidth) ---
        train_handles = [
            Patch(facecolor='white', edgecolor='black', linewidth=1.0, label='Used during Training'),
            Patch(facecolor='white', edgecolor='black', linewidth=3.0, label='Not used during Training')
        ]
        train_leg = ax.legend(handles=train_handles, title='Train status',
                              bbox_to_anchor=(1.02, 0.3), loc='upper right')
        ax.add_artist(train_leg)  # add manually so it doesn't get replaced

        # --- Pair legend ---
        # Build base palette: one color per half
        pair_handles = []
        for i, p in enumerate(pair_order):
            left_color = base_palette[i]  # left half
           # right_color = base_palette[i * 2 + 1]  # right half
            left_patch = Rectangle((0, 0), 1, 1, facecolor=left_color, edgecolor='black', label=p)
            #right_patch = Rectangle((0, 0), 1, 1, facecolor=right_color, edgecolor='black', label=f"{p} right")
            pair_handles.append(left_patch)

        # Then make legend
        pair_leg = ax.legend(handles=pair_handles, title=r"Input (Neutral or Factual $\to$ Orthogonal)",
                             bbox_to_anchor=(0.9, 1), loc='upper left', ncol=1)
        ax.add_artist(pair_leg)

        # label legend
        # --- Special label legend (yellow dot) ---
        dot_handle = Line2D([0], [0],
                            marker='o',
                            color='w',
                            label='Label',
                            markerfacecolor='yellow',
                            markeredgecolor='black',
                            markersize=10)
        dot_leg = ax.legend(handles=[dot_handle], bbox_to_anchor=(1.02, 0.7), loc='upper right')
        ax.add_artist(dot_leg)

        #plt.ylim(-1.1, 1.1)

        ax.set_xlabel(r"\gradiend Input", fontsize=15)
        ax.set_ylabel("Encoded Value $h$", fontsize=15)
        ax.set_title(f"Encoded Values {output or getattr(self, 'id', '')}")
        plt.grid(zorder=0)
        #plt.tight_layout()
        plt.show()
        if output:
            os.makedirs(os.path.dirname(output), exist_ok=True)
            plt.savefig(output)
            print(f'Saved plot to {output}')

        plt.show()
        return

    def plot_split_violin_ax(self, total_results, ax=None):
        import os
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np

        import re
        df = total_results.copy()

        # helper: parse the 'all_labels' column like in your original code
        def parse_label_value(label, label_idx):
            if isinstance(label, str):
                if label.startswith('(') and label.endswith(')'):
                    label = tuple(int(l) for l in label[1:-1].split(','))
                else:
                    label = int(label)
            if isinstance(label, tuple):
                return label[label_idx]
            else:
                # label is scalar and label_idx should be 0
                if label_idx > 0:
                    raise ValueError(f"Label index {label_idx} out of range for label {label}")
                return label


        # compute canonical (left, right, pair_label, source)
        df.loc[df['type'] == 'neutral', 'state'] = df['state']

        # map states to pretty names
        races2labelidx = self.races2labelidx.copy()
        all_races = self.all_races.copy()
        for x, y in class2pretty_name.items():
            df['state'] = df['state'].str.replace(x, y, regex=False)
            for r1, r2 in list(races2labelidx.keys()):
                if r1 == x:
                    races2labelidx[(y, r2)] = races2labelidx.pop((r1, r2))
                if r2 == x:
                    races2labelidx[(r1, y)] = races2labelidx.pop((r1, r2))
            if x in all_races:
                all_races = [r for r in all_races if r != x] + [y]
        all_races = list(sorted(all_races))

        important_states = [races2labelidx.get(x, x) for x in self.races]
        train_mask = df['source_id'].isin(important_states) & df['target_id'].isin(important_states)
        df['train'] = False
        df.loc[train_mask, 'train'] = True
        df.loc[df['source_id'] == 'train-neutral', 'source_id'] = 'neutral'

        # Build pair ordering grouped by source (left)
        # Use the order of self.all_races where possible, then neutral last if present


        df['source_id'] = df['source_id'].map(lambda x: class2pretty_name.get(x, x))
        df['target_id'] = df['target_id'].map(lambda x: class2pretty_name.get(x, x))


        unique_sources = []
        # collect sources in the order of self.all_races (if present), to keep stable ordering
        for r in all_races:
            if r in df["source_id"].values:
                unique_sources.append(r)
        if "neutral" in df["source_id"].values and "Neutral" not in unique_sources:
            unique_sources.append("Neutral")

        pair_labels = []
        sources = []
        for _, row in df.iterrows():
            raw_state = str(row["state"]).strip()
            source_id = str(row["source_id"]).strip()
            target_id = str(row["target_id"]).strip()
            # neutral handling: both GENEUTRAL and TRAIN-NEUTRAL are substates of "neutral"
            if "neutral" in raw_state.lower():
                pair_label = raw_state
                source = raw_state
            else:
                # directed form "s1->s2"
                if "->" not in raw_state:
                    raise ValueError(f"Unexpected state format (expected 's1->s2' or neutral): {raw_state}")
                s1, s2 = [p.strip() for p in raw_state.split("->", 1)]
                state_pair = (s1, s2)
                # if not present in mapping, swap order (as in your original code)
                if state_pair not in races2labelidx:
                    s1, s2 = s2, s1
                label_idx = races2labelidx[(s1, s2)]
                label_val = parse_label_value(row["all_labels"], label_idx)
                # decide canonical sorted order and arrow direction as in your original code
                states_sorted = (s1, s2)
                # your original comparison: if index(s1) < index(s2) then states_sorted=(s2,s1)
                if all_races.index(s1) < all_races.index(s2):
                    states_sorted = (s2, s1)
                if label_val == -1:
                    left, right = states_sorted[0], states_sorted[1]
                elif label_val == 1:
                    left, right = states_sorted[1], states_sorted[0]
                else:
                    raise ValueError(f"Unexpected label {label_val} for state pair {(s1, s2)}")
                pair_label = f"{source_id} → {target_id}"
                source = source_id  # we group on the left-side of the arrow
            pair_labels.append(pair_label)
            sources.append(source)
        # attach computed columns
        df = df.assign(pair=pair_labels, source=sources)

        # compute direction deterministically: 0 if matches the left substate, 1 if matches the right
        def compute_direction(row):
            raw_state = str(row["state"]).strip()
            if 'neutral' in raw_state.lower():
                return int(raw_state.lower() == 'neutral')
            # raw_state looks like s1->s2
            s1 = raw_state.split("->", 1)[0].strip()
            s2 = raw_state.split("->", 1)[1].strip()
            states_without_s1 = [s for s in all_races if s1 != s]
            return 0 if s2 == states_without_s1[0] else 1

        df["direction"] = df.apply(compute_direction, axis=1)

        # For each source, collect pair labels in deterministic order (sorted by pair_right for stability)
        pair_order = []
        source_order = list(sorted([s for s in unique_sources if s != "Neutral"])) + ['Neutral']

        for src in source_order:
            if src == 'Neutral':
                prs = [r'\traindatazero', '\gerneutral']
            else:
                prs = sorted(df.loc[df["source_id"] == src, "pair"].unique())
            pair_order.extend(prs)

        # create categorical to preserve plotting order
        df["pair"] = pd.Categorical(df["pair"], categories=pair_order, ordered=True)


        # Build base color per pair (distinct color per pair)
        n_pairs = max(1, len(pair_order))
        base_palette = sns.color_palette("Paired", n_colors=n_pairs)

        sns.violinplot(
            data=df,
            x="source_id",
            y="encoded",
            hue="direction",
            split=True,
            inner='quartile',
            order=source_order,
            ax=ax,
            density_norm="width",
            linewidth=0.7,
            #common_norm=False,
            zorder=3,
            legend=False,
        )

        # adjust linewidths correctly
        train_map = {(row.source_id, row.direction): row.train for _, row in df.iterrows()}

        hue_order = sorted(df['direction'].unique())  # same as used in violinplot
        num_x = len(source_order)

        for i, violin in enumerate(ax.collections):
            # compute which x and hue this collection corresponds to
            x_tick = i // len(hue_order)  # which source
            hue_idx = i % len(hue_order)  # which hue
            src_val = source_order[x_tick]
            dir_val = hue_order[hue_idx]

            if (src_val, dir_val) in train_map and train_map[(src_val, dir_val)]:
                violin.set_linewidth(2.5)
            else:
                violin.set_linewidth(0.5)

        # add yellow label dots
        if True:
            df['pretty_source_id'] = df['source_id'].map(lambda x: class2pretty_name.get(x, x))
            for xtick, label in enumerate(ax.get_xticklabels()):
                label_text = label.get_text()
                label_vals = df[df['pretty_source_id'] == label_text]['label'].unique()
                if len(label_vals) == 1:
                    ax.plot(xtick, label_vals[0], 'ro',
                            markersize=7, zorder=20,
                            markeredgecolor='black', markerfacecolor='yellow')

        # recolor halves
        n_halves = len(pair_order) * 2
        base_palette = sns.color_palette("Paired", n_colors=n_halves)
        for i, poly in enumerate(ax.collections):
            if i >= n_halves:
                continue
            color = base_palette[i]
            poly.set_facecolor(color)
            poly.set_edgecolor("black")
            poly.set_alpha(1.0)

        # hide xticks (global labeling is handled outside)
        ax.set_xlabel("")
        ax.set_ylabel("")

        # return elements needed for global legends
        # e.g. handles for pair legend, training legend, dot legend
        train_handles = [
            Patch(facecolor='white', edgecolor='black', linewidth=1.0, label='Not used during training'),
            Patch(facecolor='white', edgecolor='black', linewidth=3.0, label='Used during training')
        ]
        pair_handles = [
            Rectangle((0, 0), 1, 1, facecolor=base_palette[i], edgecolor='black', label=p)
            for i, p in enumerate(pair_order)
        ]
        dot_handle = Line2D([0], [0], marker='o', color='w',
                            label='Label', markerfacecolor='yellow',
                            markeredgecolor='black', markersize=7)
        return train_handles, pair_handles, dot_handle



    def print_examples_by_modes(self, df, n_examples=10):

        def canonical_pair(row):
            if row["state"] == "neutral":
                return "GENEUTRAL"
            s1, s2 = row["state"].split("->")
            state_pair = (s1, s2)
            if state_pair not in self.races2labelidx:
                s1, s2 = s2, s1
            label_idx = self.races2labelidx[(s1, s2)]
            label = row['all_labels']
            if isinstance(label, str):
                if label.startswith('(') and label.endswith(')'):
                    label = tuple(int(l) for l in label[1:-1].split(','))
                else:
                    label = int(label)
            if isinstance(label, tuple):
                label = label[label_idx]
            elif label_idx > 0:
                raise ValueError(f"Label index {label_idx} out of range for label {label}")
            states_sorted = (s1, s2)
            if self.all_races.index(s1) < self.all_races.index(s2):
                states_sorted = (s2, s1)
            if label == 1:
                return rf"{states_sorted[0]} $\to$ {states_sorted[1]}"
            elif label == -1:
                return rf"{states_sorted[1]} $\to$ {states_sorted[0]}"
            else:
                raise ValueError(f"Unexpected label {label} for state pair {state_pair}")

        df["pair"] = df.apply(canonical_pair, axis=1)
        pair_order = df['pair'].unique()

        for pair in pair_order:
            group = df[df["pair"] == pair]
            if group.empty or pair == "GENEUTRAL":
                continue

            values = group["encoded"].values

            # Build histogram to detect peaks
            hist, bin_edges = np.histogram(values, bins=40)
            peaks, _ = find_peaks(hist, distance=3)

            if len(peaks) == 0:
                # fallback: just pick min and max
                centers = [np.min(values), np.max(values)]
            else:
                # take bin centers of peaks
                centers = [(bin_edges[p] + bin_edges[p + 1]) / 2 for p in peaks]

            print('Start')
            print(f"\n=== {pair} ===")
            for center in centers:
                # find rows closest to each peak center
                group["dist"] = np.abs(group["encoded"] - center)
                closest = group.nsmallest(n_examples, "dist")

                print(f"\n--- Around mode at {center:.3f} ---")
                for _, row in closest.iterrows():
                    txt = row["text"]

                    # shorten text
                    if "[MASK]" in txt:
                        txt = (
                                "..."
                                + txt.split("[MASK]")[0][-100:].strip()
                                + f" [[{row['source']}]] "
                                + txt.split("[MASK]")[1][:100].strip()
                                + "..."
                        )
                    elif row["source"] in txt:
                        txt = (
                                "..."
                                + txt.split(row["source"])[0][-100:].strip()
                                + f' [{row["source"]}] '
                                + txt.split(row["source"])[1][:100].strip()
                                + "..."
                        )
                    else:
                        txt = row["source"]

                    txt = "".join(c if 32 <= ord(c) <= 126 else "?" for c in txt)
                    print(f"{row['encoded']:.3f}: {txt}")



    def post_training(self, model_with_gradiend, **kwargs):
        pass

    def plot_model_selection(self, model_with_gradiend):
        data = self.evaluate_gradiend(model_with_gradiend)
        model_name = model_with_gradiend.name_or_path
        c1, c2 = self.races
        pretty_classes = {

        }
        pc1, pc2 = pretty_classes.get(c1, c1), pretty_classes.get(c2, c2)
        swapped = False
        if pc1 > pc2:
            c1, c2 = c2, c1
            swapped = True

        metrics = [f'bias_{self.id}->group_probs->{c1}', f'bias_{self.id}->group_probs->{c2}', 'lms->lms', self.id]
        highlight_metrics = [self.id]
        plot_gradiend_model_selection(data,
                                      model_name,
                                      metrics=metrics,
                                      highlight_metrics=highlight_metrics,
                                      mirror_lr_axis=swapped,
                                      plot_numbers=True,
                                      linear_plot=False,
                                      horizontal_plot=True,
                                      )


    def get_model_selection_stats(self, model_with_gradiend):
        data = self.evaluate_gradiend(model_with_gradiend)
        return data

    def evaluate_gradiend(self,
                          model_with_gradiend,
                          #feature_factors=(0.0, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.8, 0.9, 1.0),
                          #feature_factors=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0),
                          feature_factors=(-10.0, -2.0, -1.0, -0.8, -0.6, -0.4, -0.2, -0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 10.0),
                          lrs=(-5e-1, -1e-1, -5e-2, -1e-2, -5e-3, -1e-3, -5e-4, -1e-4, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1),
                          top_k=None,
                          part='decoder',
                          top_k_part='decoder',
                          **kwargs,
                          ):
        path = model_with_gradiend.name_or_path

        base_model = model_with_gradiend.base_model
        tokenizer = model_with_gradiend.tokenizer
        model_id = os.path.basename(path) if path.startswith('results/models') else path
        base_file, file = get_evaluation_file(f'results/cache/decoder/{self.id}/{model_id}', feature_factors, lrs, top_k=top_k, part=part, top_k_part=top_k_part)
        os.makedirs(os.path.dirname(base_file), exist_ok=True)
        os.makedirs(os.path.dirname(file), exist_ok=True)

        pairs = {(feature_factor, lr) for feature_factor in feature_factors for lr in lrs}
        # randomize
        import random
        pairs = list(pairs)
        random.shuffle(pairs)
        #pairs = list(sorted(pairs, key=lambda x: (abs(x[0]), -abs(x[1]), x[0], x[1])))
        expected_results = len(pairs) + 1 + (len(self.metric_keys))  # 1 because of base

        try:
            relevant_results = json.load(open(file, 'r'))

            raw_relevant_results = convert_results_to_dict(relevant_results)
            relevant_results = {}
            for k, v in raw_relevant_results.items():
                relevant_results[k] = v

            # check if complete
            if len(relevant_results) == expected_results:
                return relevant_results
        except FileNotFoundError:
            relevant_results = {}
        except Exception as e:
            print(f'Error for {file}')
            raise e

        try:
            all_results = json.load(open(base_file, 'r'))
            raw_all_results = convert_results_to_dict(all_results)

            all_results = raw_all_results.copy()

            # copy relevant results into relevant_results
            for pair in pairs:
                if pair in all_results:
                    relevant_results[pair] = all_results[pair]

            if 'base' in all_results:
                relevant_results['base'] = all_results['base']

            if len(relevant_results) == expected_results:
                with open(file, 'w+') as f:
                    json.dump(convert_results_to_list(relevant_results), f, indent=2)
                return relevant_results

        except FileNotFoundError:
            all_results = {}

        if 'base' in relevant_results:
            print("Skipping base model as it is already evaluated")
        else:
            base_results = self.evaluate_model(base_model, tokenizer, force=True)
            all_results['base'] = base_results
            relevant_results['base'] = base_results

        for feature_factor, lr in tqdm(pairs, desc=f"Evaluate GRADIEND", total=len(pairs)):
            id = {'feature_factor': feature_factor, 'lr': lr}
            id_key = (feature_factor, lr)
            if id_key in relevant_results:
                print(f"Skipping {id} as it is already evaluated")
                continue

            enhanced_model = model_with_gradiend.modify_model(lr=lr, feature_factor=feature_factor, top_k=top_k, part=part, top_k_part=top_k_part)

            enhanced_bert_results = self.evaluate_model(enhanced_model, tokenizer, cache_folder=f'{feature_factor}_{lr}', model_id=model_id)
            all_results[id_key] = enhanced_bert_results
            relevant_results[id_key] = enhanced_bert_results

            score = enhanced_bert_results[self.id]
            print(f"Evaluated {id_key} with score {score}")

            with open(base_file, 'w+') as f:
                json.dump(convert_results_to_list(all_results), f, indent=2)

            # free memory
            del enhanced_model
            torch.cuda.empty_cache()


        list_results = convert_results_to_list(relevant_results)
        with open(file, 'w+') as f:
            json.dump(list_results, f, indent=2)
        return convert_results_to_dict(list_results)



    def evaluate_model(self, model, tokenizer, force=False, cache_folder='', model_id=None, verbose=False, base_model_stats=None, **additional_stats):
        model_hash = hash_model_weights(model)
        model_name = model.name_or_path
        model_id = model_id or os.path.basename(model_name)
        if cache_folder and not cache_folder.endswith('/'):
            cache_folder += '/'

        cache_file = f'results/cache/evaluate_model/{self.id}/{model_id}/{cache_folder}{model_hash}.json'
        if False and not force:
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except FileNotFoundError or json.JSONDecodeError:
                pass

        max_size = 1000
        result = self._evaluate_model(model, tokenizer, max_size=max_size)

        lms = result['lms']['lms']
        for key in [self.id]:
            score = result[f'bias_{key}']['score']
            result[key] = score * lms

        if additional_stats:
            result['stats'] = additional_stats

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w+') as f:
            json.dump(result, f, indent=2)

        return result

    def read_data(self, tokenizer, max_size=None, split=None, races=None):
        races = races or self.races

        id = f'{tokenizer.name_or_path}_{self.id}_{max_size}_{split}'

        if id in self.data_cache:
            return self.data_cache[id]


        data = create_training_dataset(tokenizer, max_size, batch_size=1, split=split, races=races, bias_type= self.bias_type)
        self.data_cache[id] = data

        return data

    def _evaluate_model(self, model, tokenizer, max_size=None):
        df = self.read_data(tokenizer, max_size=max_size, split='test') # todo
        df = df.data

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # 2. compute lms
        start_time = time.time()
        lms = compute_lms(model, tokenizer, df['text'], ignore=self.non_neutral_terms)
        print(f"Computed LMs for {len(df)} samples in {time.time() - start_time:.2f} seconds")

        # 1. compute bias score
        start = time.time()
        bias_score = compute_bias_score(model, tokenizer, df, key_text='masked', targets=self.targets)
        print(f"Computed bias score for {len(df)} samples in {time.time() - start:.2f} seconds")

        return {
            f'bias_{self.id}': bias_score,
            'lms': lms,
        }

    def _evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        score = result['score']
        encoded = result['encoded']
        encoded_by_class = result['encoded_by_class']
        mean_by_class = result['mean_by_class']

        x = []
        y = []
        for label, encoded_values in encoded_by_class.items():
            x.extend(encoded_values)
            parsed_label = label if isinstance(label, (int, float)) else float(label) if isinstance(label, (bool, np.bool_)) else float(label[1:-2])  # Handle tuple string labels
            y.extend([parsed_label] * len(encoded_values))

        y_binary = [1 if val > 0 else 0 for val in y]
        # plot with color based on y_binary
        cmap = ListedColormap(['blue', 'red']) or 'coolwarm'
        plt.scatter(x, y, c=y_binary, cmap=cmap, alpha=1.0)

        counterfactual_targets = result['counterfactual_target']
        # plot counterfactual targets as text labels
        #for i, target in enumerate(counterfactual_targets):
        #    plt.text(x[i][0], y[i], target, fontsize=6, alpha=0.7)

        plt.title(f"{self.target_key} Evaluation - Score: {score}")
        plt.xlabel('Encoded Feature')
        plt.ylabel(f'{self.target_key} Value')
        plt.grid()
        output = f'img/gradiend/emotion_{self.target_key}_evaluation_{self.ctr}_{score}.png'
        os.makedirs(os.path.dirname(output), exist_ok=True)
        plt.savefig(output)
        self.ctr += 1
        plt.show()

        return result

class Bias1DSetup(BiasSetup):
    def __init__(self, bias_type, race1, race2, pretty_id=None):
        super().__init__(bias_type, race1, race2, pretty_id=pretty_id)
        self.n_features = 1

    @property
    def race1(self):
        return self.races[0]

    @property
    def race2(self):
        return self.races[1]

class Race1DSetup(Bias1DSetup):
    def __init__(self, race1, race2, pretty_id=None):
        super().__init__('race', race1, race2, pretty_id=pretty_id)

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


class Bias3DCombinedSetup(BiasSetup):
    def __init__(self, bias_type, *types):
        super().__init__(bias_type, *types)
        self.races = list(types)
        self.short_id = self.id
        self.id += '_combined'
        self.metric_keys = [self.id]
        self.n_features = 3


    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(*args, races=self.races, bias_type=self.bias_type, **kwargs)


    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        encoded = result['encoded']
        labels = result['labels']

        import matplotlib.pyplot as plt
        import numpy as np

        encoded = np.array(result['encoded'])  # shape (n, 3)
        labels = np.array(result['labels'])  # shape (n,)

        # Map labels to colors and names
        label_to_color = {-1: "red", 0: "blue", 1: "green"}
        label_to_name = {-1: "Class -1", 0: "Class 0", 1: "Class 1"}
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D

        encoded = np.array(result['encoded'])  # (n, 3)
        labels = np.array(result['labels'])  # (n, 3)

        # Turn each row of labels into a tuple so we can color uniquely
        combined_labels = [tuple(l) for l in labels]

        # Assign a color to each unique tuple
        unique_labels = list(set(combined_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {lab: colors[i] for i, lab in enumerate(unique_labels)}

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for lab in unique_labels:
            idx = np.array(combined_labels) == lab
            idx = np.all(labels == lab, axis=1)  # mask for this tuple
            ax.scatter(encoded[idx, 0], encoded[idx, 1], encoded[idx, 2],
                       c=[label_to_color[lab]], label=str(lab), s=50, alpha=0.7, edgecolor="k")

        ax.set_xlabel("Encoding[0]")
        ax.set_ylabel("Encoding[1]")
        ax.set_zlabel("Encoding[2]")
        ax.set_title("3D Encodings with 3D Labels")
        ax.legend()
#        output = f'results/experiments/'

        plt.show()

        return result

    def evaluate_gradiend__(self,
                          model_with_gradiend,
                          #feature_factors=(-1.0, -0.5, 0.0, 0.5, 1.0),
                          #lrs=(-1e-3, -1e-2, -1e-1, 1e-3, 1e-2, 1e-1),
                          #feature_factors=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0),
                          #feature_factors=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0, -0.2, -0.4, -0.6, -0.8, -1.0),
                          feature_factors=(0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0),
                          lrs=(-1e-2, -1e-1, 1e-2, 1e-1),
                          top_k=None,
                          part='decoder',
                          top_k_part='decoder',
                          plot=True,
                          **kwargs,
                          ):
        path = model_with_gradiend.name_or_path

        base_model = model_with_gradiend.base_model
        tokenizer = model_with_gradiend.tokenizer
        model_id = os.path.basename(path) if path.startswith('results/models') else path
        base_file, file = get_evaluation_file(f'results/cache/decoder/{self.id}/{model_id}', feature_factors, lrs, top_k=top_k, part=part, top_k_part=top_k_part)
        os.makedirs(os.path.dirname(base_file), exist_ok=True)
        os.makedirs(os.path.dirname(file), exist_ok=True)

        # blow up feature_factors s.t. it contains all combinations of 3 factors
        feature_factors = list(itertools.product(feature_factors, repeat=3))

        pairs = {(tuple(feature_factor), lr) for feature_factor in feature_factors for lr in lrs}
        expected_results = len(pairs) + 1 + (len(self.metric_keys))  # 1 because of base

        try:
            relevant_results = json.load(open(file, 'r'))

            raw_relevant_results = convert_results_to_dict(relevant_results)
            relevant_results = {}
            for k, v in raw_relevant_results.items():
                relevant_results[k] = v

            # check if complete
            if len(relevant_results) == expected_results:
                return relevant_results
        except FileNotFoundError:
            relevant_results = {}
        except Exception as e:
            print(f'Error for {file}')
            raise e

        try:
            all_results = json.load(open(base_file, 'r'))
            raw_all_results = convert_results_to_dict(all_results)

            all_results = raw_all_results.copy()

            # copy relevant results into relevant_results
            for pair in pairs:
                if pair in all_results:
                    relevant_results[pair] = all_results[pair]

            if 'base' in all_results:
                relevant_results['base'] = all_results['base']

            if len(relevant_results) == expected_results:
                with open(file, 'w+') as f:
                    json.dump(convert_results_to_list(relevant_results), f, indent=2)
                return relevant_results

        except FileNotFoundError:
            all_results = {}

        if 'base' in relevant_results:
            print("Skipping base model as it is already evaluated")
        else:
            base_results = self.evaluate_model(base_model, tokenizer, force=True)
            all_results['base'] = base_results
            relevant_results['base'] = base_results

        for feature_factor, lr in tqdm(pairs, desc=f"Evaluate GRADIEND {path}"):
            id = {'feature_factor': feature_factor, 'lr': lr}
            id_key = (feature_factor, lr)
            if id_key in relevant_results:
                print(f"Skipping {id} as it is already evaluated")
                continue

            enhanced_model = model_with_gradiend.modify_model(lr=lr, feature_factor=feature_factor, top_k=top_k, part=part, top_k_part=top_k_part)

            enhanced_bert_results = self.evaluate_model(enhanced_model, tokenizer, cache_folder=f'{feature_factor}_{lr}')
            all_results[id_key] = enhanced_bert_results
            relevant_results[id_key] = enhanced_bert_results

            score = enhanced_bert_results[self.id]
            print(f"Evaluated {id_key} with score {score}")

            with open(base_file, 'w+') as f:
                json.dump(convert_results_to_list(all_results), f, indent=2)

            # free memory
            del enhanced_model
            torch.cuda.empty_cache()


        list_results = convert_results_to_list(relevant_results)
        with open(file, 'w+') as f:
            json.dump(list_results, f, indent=2)
        dict_results = convert_results_to_dict(list_results)


        if plot:
            plot_gradiend_model_selection(dict_results, model_id, feature_factors=feature_factors, lrs=lrs,
                                          metrics=[self.id],
                                          **kwargs)

        return dict_results



    def evaluate_gradiend(self,
                          model_with_gradiend,
                          feature_factors=(0.0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75, -1.0),
                          lrs=(-1e-3, -1e-2, -1e-1, 1e-3, 1e-2, 1e-1),
                          top_k=None,
                          part='decoder',
                          top_k_part='decoder',
                          plot=False, # plotting not supported for multidimensional features!
                          search_method="bayes",  # grid, heuristic, bayes
                          n_samples=20,  # heuristic: number of samples
                          n_calls=200,  # bayes: number of evaluations
                          **kwargs,
                          ):
        path = model_with_gradiend.name_or_path

        base_model = model_with_gradiend.base_model
        tokenizer = model_with_gradiend.tokenizer
        model_id = os.path.basename(path) if path.startswith('results/models') else path
        base_file, file = get_evaluation_file(
            f'results/cache/decoder/{self.id}/{model_id}', feature_factors, lrs,
            top_k=top_k, part=part, top_k_part=top_k_part
        )
        os.makedirs(os.path.dirname(base_file), exist_ok=True)
        os.makedirs(os.path.dirname(file), exist_ok=True)

        # blow up feature_factors s.t. it contains all combinations of 3 factors
        feature_factors = list(itertools.product(feature_factors, repeat=3))
        all_pairs = [(tuple(feature_factor), lr) for feature_factor in feature_factors for lr in lrs]

        if search_method == "grid":
            pairs = set(all_pairs)

        elif search_method == "heuristic":
            pairs = set(random.sample(all_pairs, min(n_samples, len(all_pairs))))

        elif search_method == "bayes":
            # Define continuous spaces for Bayesian optimization
            space = [
                Real(-10.0, 10.0, name="f1"),
                Real(-10.0, 10.0, name="f2"),
                Real(-10.0, 10.0, name="f3"),
                Real(1e-5, 1e-1, prior="log-uniform", name="lr_mag"),  # magnitude
                Categorical([-1, 1], name='lr_sign')
            ]

            evaluated = {}

            pbar = tqdm(total=n_calls)
            @use_named_args(space)
            def objective(**params):
                f1, f2, f3 = params['f1'], params['f2'], params['f3']
                lr_mag = params['lr_mag']
                lr_sign = params['lr_sign']
                feature_factor = (f1, f2, f3)
                lr = lr_sign * lr_mag
                id_key = (feature_factor, lr)

                if id_key in evaluated:
                    return evaluated[id_key]

                enhanced_model = model_with_gradiend.modify_model(
                    lr=lr, feature_factor=feature_factor,
                    top_k=top_k, part=part, top_k_part=top_k_part
                )
                results = self.evaluate_model(enhanced_model, tokenizer, cache_folder=f'{feature_factor}_{lr}')
                score = results[self.id]
                evaluated[id_key] = -score  # skopt minimizes
                pbar.update(1)
                pbar.set_postfix({"best": -min(evaluated.values()), "current": -score, 'lms': results['lms']['lms']})
                return -score

            res = gp_minimize(objective,
                              space,
                              n_calls=n_calls,
                              random_state=0,
                              )

            # Plot convergence curve (objective over iterations)
            base_output = f'results/models/{self.id}/{model_id}/skopt_{n_calls}'
            plot_convergence(res)
            plt.savefig(f'{base_output}_convergence.pdf')

            plt.show()

            # Pairwise dependencies and objective landscape
            plot_objective(res)
            plt.savefig(f'{base_output}_objective.pdf')
            plt.show()

            # Distribution of sampled points in the search space
            plot_evaluations(res)
            plt.savefig(f'{base_output}_evaluations.pdf')
            plt.show()

            res.specs["args"]["func"] = None  # remove function for pickling
            dump(res, f'{base_output}.pkl')

            # Convert evaluated dict into pairs set
            #pairs = set(evaluated.keys())

            # get best pair
            best_pair = min(evaluated, key=evaluated.get)
            print(f"Best pair: {best_pair} with score {-evaluated[best_pair]}")
            pairs = {best_pair}

        else:
            raise ValueError(f"Unknown search_method: {search_method}")

        expected_results = len(pairs) + 1 + len(self.metric_keys)  # 1 for base

        try:
            relevant_results = json.load(open(file, 'r'))
            raw_relevant_results = convert_results_to_dict(relevant_results)
            relevant_results = {k: v for k, v in raw_relevant_results.items()}
            if len(relevant_results) == expected_results:
                return relevant_results
        except FileNotFoundError:
            relevant_results = {}
        except Exception as e:
            print(f'Error for {file}')
            raise e

        try:
            all_results = json.load(open(base_file, 'r'))
            raw_all_results = convert_results_to_dict(all_results)
            all_results = raw_all_results.copy()
            for pair in pairs:
                if pair in all_results:
                    relevant_results[pair] = all_results[pair]
            if 'base' in all_results:
                relevant_results['base'] = all_results['base']
            if len(relevant_results) == expected_results:
                with open(file, 'w+') as f:
                    json.dump(convert_results_to_list(relevant_results), f, indent=2)
                return relevant_results
        except FileNotFoundError:
            all_results = {}

        if 'base' in relevant_results:
            print("Skipping base model as it is already evaluated")
            base_results = relevant_results['base']
        else:
            base_results = self.evaluate_model(base_model, tokenizer, force=True)
            all_results['base'] = base_results
            relevant_results['base'] = base_results

        for feature_factor, lr in tqdm(pairs, desc=f"Evaluate GRADIEND", total=len(pairs)):
            id = {'feature_factor': feature_factor, 'lr': lr}
            id_key = (feature_factor, lr)
            if id_key in relevant_results:
                print(f"Skipping {id} as it is already evaluated")
                continue

            enhanced_model = model_with_gradiend.modify_model(
                lr=lr, feature_factor=feature_factor,
                top_k=top_k, part=part, top_k_part=top_k_part
            )
            enhanced_bert_results = self.evaluate_model(enhanced_model, tokenizer,
                                                        cache_folder=f'{feature_factor}_{lr}')
            all_results[id_key] = enhanced_bert_results
            relevant_results[id_key] = enhanced_bert_results

            score = enhanced_bert_results[self.id]
            current_best = max([v[self.id] for v in relevant_results.values() if isinstance(v, dict)] + [base_results[self.id]])
            print(f"Evaluated {id_key} with score {score} (best: {current_best})")

            with open(base_file, 'w+') as f:
                json.dump(convert_results_to_list(all_results), f, indent=2)

            del enhanced_model
            torch.cuda.empty_cache()


        list_results = convert_results_to_list(relevant_results)
        with open(file, 'w+') as f:
            json.dump(list_results, f, indent=2)
        dict_results = convert_results_to_dict(list_results)

        if plot:
            plot_gradiend_model_selection(dict_results, model_id, feature_factors=feature_factors, lrs=lrs,
                                          metrics=[self.id], **kwargs)

        return dict_results


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

class Race3DSetup(BiasSetup):
    def __init__(self):
        super().__init__('race', 'white', 'black', 'asian')
        self.n_features = 3


    def create_training_data(self, *args, **kwargs):
        return create_training_dataset(races=self.races, *args, **kwargs)



def train_race_gradiends(configs, version=None, activation='tanh', setups=None):

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
        #'gpt2': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff'),
        #'roberta-large': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=4),
        #'distilbert-base-cased': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff'),
        #'bert-base-cased': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=8),
        #'bert-large-cased': dict(eval_max_size=500, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=4),

        'meta-llama/Llama-3.2-3B-Instruct': dict(eval_max_size=50, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=1, torch_dtype=torch.bfloat16, lr=1e-4),
        #'meta-llama/Llama-3.2-3B': dict(eval_max_size=50, batch_size=32, n_evaluation=250, epochs=20, max_iterations=2500, source='factual', target='diff', eval_batch_size=1, torch_dtype=torch.bfloat16, lr=1e-4),
    }




    configs_factual_source = {k: {**v, 'source': 'factual'} for k, v in configs.items()}

    all_setups = [
        #ChristianJewishSetup(),
        #MuslimJewishSetup(),
        #WhiteAsianSetup(),
        BlackAsianSetup(),
        #WhiteBlackSetup(),
        #ChristianMuslimSetup(),
    ]

#    all_setups = list(reversed(all_setups))

    try:
        train_race_gradiends(configs, version='v5', activation='tanh', setups=all_setups)
    except NotImplementedError as e:
        print(f"Error during training: {e}")



