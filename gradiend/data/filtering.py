import json
import os.path
import pickle
import time
import re

from nltk.tokenize import sent_tokenize
# Download necessary data for nltk
#nltk.download('punkt')

import numpy as np
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd

from gradiend.data import read_bookcorpus, read_gender_data, gender_pronouns, read_processed_bookscorpus, read_namexact, \
    read_namextend
from gradiend.data.split import split
from gradiend.model import AutoTokenizerForLM
from gradiend.util import evaluate_he_she



label_to_pronoun = {
    'M': 'he',
    'F': 'she'
}

invert_label = {
    'M': 'F',
    'F': 'M',
    'B': 'B'
}

pronouns_pairs = {
    ('he', 'she'), # subjective pronoun
    ('him', 'her'), # objective pronoun
    ('his', 'her/hers'), # possessive pronoun
    ('himself', 'herself'), # reflexive pronoun
}

def contains_he(text):
    return bool(re.search(r'he', text, flags=re.IGNORECASE))

class GenderTextFilter:

    def __init__(self, names, all_names, text='text'):
        self.names = names
        self.all_names_df = all_names
        self.text = text
        self.names['name_lower'] = self.names['name'].str.lower()
        self.all_names_df['name_lower'] = self.all_names_df['name'].str.lower()
        # some names occur multiple times as male and female names
        self.all_relevant_names = self.names['name'].unique()
        self.all_names = self.all_names_df['name'].unique()

        self.groups = {f'G{i}': k for i, k in enumerate(self.all_relevant_names)}
        self.all_groups = {f'G{i}': k for i, k in enumerate(self.all_names)}

        grouped_names = '|'.join(f'(?P<{group_name}>{re.escape(name.lower())})' for group_name, name in self.groups.items())
        grouped_all_names = '|'.join(f'(?P<{group_name}>{re.escape(name.lower())})' for group_name, name in self.all_groups.items())
        self.pattern = re.compile(r'\b(' + grouped_names + r')\b', re.IGNORECASE)
        self.pattern_all_names = re.compile(r'\b(' + grouped_all_names + r')\b', re.IGNORECASE)

        self.pattern_he = re.compile(r'\b(' + 'he' +  r')\b', re.IGNORECASE)
        self.pattern_she = re.compile(r'\b(' + 'she' +  r')\b', re.IGNORECASE)

        self.names_to_label = dict(self.names.drop_duplicates('name_lower').set_index('name_lower')['genders'])
        self.names_to_label.update(self.names.drop_duplicates('name').set_index('name')['genders'])

        self.gendered_words = read_gender_data()
        male_pronouns = ['him', 'his', 'himself']
        female_pronouns = ['her', 'hers', 'herself']
        self.pattern_gender_pronouns = re.compile(r'\b(' + '|'.join(gender_pronouns) + r')\b')
        self.pattern_male_pronouns = re.compile(r'\b(' + '|'.join(male_pronouns) + r')\b', re.IGNORECASE)
        self.pattern_female_pronouns = re.compile(r'\b(' + '|'.join(female_pronouns) + r')\b', re.IGNORECASE)
        self.pattern_pronouns = re.compile(r'\b(' + '|'.join(female_pronouns + male_pronouns) + r')\b', re.IGNORECASE)

        male_words = self.gendered_words['M']
        female_words = self.gendered_words['F']
        self.pattern_male_words = re.compile(r'\b(' + '|'.join(male_words) + r')\b', re.IGNORECASE)
        self.pattern_female_words = re.compile(r'\b(' + '|'.join(female_words) + r')\b', re.IGNORECASE)
        self.pattern_gender_words = re.compile(r'\b(' + '|'.join(female_words + male_words) + r')\b', re.IGNORECASE)

        self.forbidden_sequences = ['``', '"']

        self.ctr = 0

    def filter_entry(self, entry):
        entry['masked'] = ''
        entry['label'] = ''
        entry['name'] = ''
        entry['pronoun'] = ''
        entry['other_gender_pronouns'] = False
        entry['gender_words'] = False
        entry['pronoun_count'] = 0
        return entry

    def __call__(self, entry):
        text = entry[self.text]

        if len(text) < 50:
            return self.filter_entry(entry)

        if re.search('|'.join(self.forbidden_sequences), text):
            return self.filter_entry(entry)

        # check if he is in the text (note she => he)
        if not contains_he(text):
            return self.filter_entry(entry)

        first_match = None
        for match in self.pattern.finditer(text):
            if first_match is None:
                first_match = match
            else:
                return self.filter_entry(entry) # we found multiple names, so return

        if first_match is None:
            return self.filter_entry(entry) # we found no names, so return

        # check if there are any match of a dataset with more names, that potentially are used with different meeaning here
        def re_match_eq(m1, m2):
            return m1.span() == m2.span()

        for match in self.pattern_all_names.finditer(text):
            if not re_match_eq(match, first_match):
                # multiple names occur
                return self.filter_entry(entry)

        match = first_match
        start, end = match.span()
        matched_name = text[start:end]
        subsequent_text = text[end:]
        label = self.names_to_label[matched_name.lower()]
        masked_text = text[:start] + '[NAME]' + subsequent_text
        pronoun = label_to_pronoun.get(label)

        def check_match(match):
            # filter those matches that are before the name
            match = [m for m in match if m.span()[1] >= start]
            # if he/she occurs at least once after the name, its a valid match!
            return len(match) > 0


        # check if he/she comes after the name, according to the gender
        if label == 'B':
            # in contrast to F and M, we dont know what pronoun to expect here, so we collect all we can find (both he and she)
            # and see if all found agree on a gender
            match_he = list(re.finditer(self.pattern_he, masked_text))
            match_she = list(re.finditer(self.pattern_she, masked_text))

            if match_he:
                if match_she:
                    return self.filter_entry(entry) # he and she found -> we can not be sure which gender is used, so we ignore this entry
                match_pronoun = match_he
                pronoun = 'he'
            elif match_she:
                match_pronoun = match_she
                pronoun = 'she'
            else:
                return self.filter_entry(entry)

        elif label in ['F', 'M']:
            if label == 'F':
                pattern_pronoun = self.pattern_she
                pattern_inverse_pronoun = self.pattern_he
            elif label == 'M':
                pattern_pronoun = self.pattern_he
                pattern_inverse_pronoun = self.pattern_she

            match_pronoun = list(re.finditer(pattern_pronoun, masked_text))
            if not match_pronoun:
                return self.filter_entry(entry)

            # check if the other pronoun also occurs
            match_inverse_pronoun = list(re.finditer(pattern_inverse_pronoun, masked_text))
            if match_inverse_pronoun:
                # the other pronoun also occurs, -> this might lead to ambiguity when replacing the name with a name of different gender
                return self.filter_entry(entry)
        else:
            return self.filter_entry(entry)

        # Iterate over all matches (reversed to avoid false indices)
        for match in reversed(match_pronoun):
            s, e = match.span()
            masked_text = masked_text[:s] + '[PRONOUN]' + masked_text[e:]
            if e < end:
                return self.filter_entry(entry)

        # assume the other pronoun does not occur
        if label != 'B':
            inverted_label = invert_label[label]
            inverted_pronoun = label_to_pronoun[inverted_label]
            pattern = r'\b' + inverted_pronoun + f'\b'
            # Use re.search to check for the presence of the pattern
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self.filter_entry(entry)

        entry['masked'] = masked_text
        entry['label'] = label
        entry['name'] = matched_name
        entry['pronoun'] = pronoun
        entry['pronoun_count'] = len(match_pronoun)

        # label whether there are other pronouns (him/her/himself/herself)
        # note that this might lead to noisy labels
        # additionally, label whether a gender specific word is used (might also indicate noise)
        if pronoun in ['he', 'she']:
            gender_pronouns = bool(re.search(self.pattern_pronouns, masked_text))
            gender_words = bool(re.search(self.pattern_gender_words, masked_text))
        else:
            gender_pronouns = False
            gender_words = False

        entry['other_gender_pronouns'] = gender_pronouns
        entry['gender_words'] = gender_words

        self.ctr += 1
        print(f'{self.ctr}: {masked_text} -> [NAME] = {matched_name}, [PRONOUN] = {pronoun}')
        return entry





# filters text such that only texts without any gender specific word occurrence is contained
class NoGenderTextFilter:

    def __init__(self, all_names, text='text'):
        self.all_names_df = all_names
        self.text = text
        self.all_names_df['name_lower'] = self.all_names_df['name'].str.lower()
        self.all_names = self.all_names_df['name'].unique().tolist()

        # names + gender words + pronouns +
        pronouns = ['he', 'she', 'him', 'her', 'his', 'hers', 'himself', 'herself']
        self.gendered_words = read_gender_data()
        self.gendered_words = [word for words in self.gendered_words for word in words]

        gender_words = self.all_names + pronouns + self.gendered_words
        self.pattern = re.compile(r'\b(' + '|'.join(gender_words) + r')\b', re.IGNORECASE)

    def __call__(self, entry):
        text = entry[self.text]

        if len(text) < 50:
            return False

        return bool(re.search(self.pattern, text))

def get_names_datasets():
    df = read_namexact()

    # we also need a more broader dataset containing all names (also those that might have different meanings) to filter out false positive sentences
    # we use all names that occur at least 100 times to avoid very rare names (this reduces number of names by more than ~40k to ~90k
    # and filters out e.g. the name Had which would exclude all name with the verb "had" if considered as name here)
    df_all_names = read_namextend()

    return df, df_all_names

def filter_gender_text():
    df, df_all_names = get_names_datasets()

    # Load the Wikipedia dataset (latest version)
    dataset = read_bookcorpus()

    processed_bookscorpus = read_processed_bookscorpus()
    text_set = set(processed_bookscorpus['text']) # create set for a fast lookup
    # todo rerun the generation process, this time with the entire processed bookscorpus V1 (including gender_pronouns...)
    #filtered_dataset = dataset.filter(lambda row: row['text'] in text_set)

    #dataset = Dataset.from_pandas(read_wikipedia_gender_data())
    #dataset.to_csv('data/wiki.csv', index=False)n

    # note that wikipedia data seems to be prune to more errors than the bookscorpus dataset, as typically last names
    # are used to write about persons, so no gender information is available
    #dataset = Dataset.from_csv('data/wiki/64_100000.csv')
    start = time.time()

    batch_size = 100
    #dataset = filtered_dataset

    n = len(dataset) // batch_size
    for i, start in enumerate(range(0, len(dataset), batch_size)):
        start_time = time.time()
        output = f'data/bookscorpus/gender_data_{n}_{i}.csv'
        #output = f'data/wiki_gender_data/wiki_{n}_{i}.csv'
        if os.path.isfile(output):
            print(f'{i}/{n}: File {output} already exists')
            #continue

        current_dataset = dataset.select(range(start, min(start + batch_size, len(dataset))))
        filtered_dataset = current_dataset.map(GenderTextFilter(df, df_all_names), num_proc=1)
        end = time.time()
        print(f'{i}/{n}: Mapping took {end - start_time}s')
        filtered_dataset = filtered_dataset.filter(lambda row: row['masked'])

        print(f'Reduced dataset from {len(current_dataset)} to {len(filtered_dataset)} in {time.time() - start_time}s')
        filtered_dataset.to_csv(output)


def filter_no_gender_text():
    #df_all_names = read_names_data(filter_non_unique=False, minimum_count=10000, filter_non_name_words=False, filter_excluded_words=False, max_entries=None)
    df_all_names = read_namextend()
    # Load the Wikipedia dataset (latest version)
    dataset = read_bookcorpus()

    batch_size = 100000

    n = len(dataset) // batch_size
    files = []
    for i, start in enumerate(range(0, len(dataset), batch_size)):

        start_time = time.time()
        output = f'data/bookscorpus_no_gender/no_gender_data_{n}_{i}.csv'
        files.append(output)
        if os.path.isfile(output):
            print(f'{i}/{n}: File {output} already exists')
            continue

        current_dataset = dataset.select(range(start, min(start + batch_size, len(dataset))))
        filtered_dataset = current_dataset.filter(NoGenderTextFilter(df_all_names), num_proc=25)
        end = time.time()
        print(f'{i}/{n}: Filtering took {end - start_time}s')

        print(f'Reduced dataset from {len(current_dataset)} to {len(filtered_dataset)} in {time.time() - start_time}s')
        filtered_dataset.to_csv(output)

    # concatenate all files
    filtered_dataset = concatenate_datasets([Dataset.from_csv(file) for file in files]).to_pandas()
    filtered_dataset = filtered_dataset.drop_duplicates(['text']).reset_index(drop=True)
    filtered_dataset = filtered_dataset.sample(frac=1.0) # randomize order
    filtered_dataset.to_csv('data/geneutral.csv', index=False)

def filter_by_model_confidence(model='bert-base-uncased', threshold=1.0, df=None, df_name=None, n=50):
    if df is None:
        # use default GENTER data
        df = read_processed_bookscorpus(threshold=None)
    else:
        assert df_name is not None, 'If a custom dataframe is provided, the name of the dataframe must be provided as well'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizerForLM.from_pretrained(model)
    mask_token = tokenizer.mask_token
    model = AutoModelForMaskedLM.from_pretrained(model).to(device)


    names_df = read_namexact(split='train')
    df_name = df_name or 'genter'

    # retrieve the most common top n names per gender in the train names split
    names = names_df[names_df['genders'] != 'B'].sort_values(by='count', ascending=False).groupby('gender').apply(lambda x: x.head(n)).reset_index(drop=True)
    print(f'Most common {n} names per gender in the train split:')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(names)
    modified = False

    masked_results = {}

    model_name = model.name_or_path.replace('/', '_')
    cache_file = f'data/cache/default_predictions_train_{df_name}_{model_name}.pickle'
    cache_correct_predictions_file = f'data/cache/default_predictions_train_{df_name}_{model_name}_correct_predictions_{n}.json'
    try:
        default_predictions_dict = pickle.load(open(cache_file, 'rb'))
    except FileNotFoundError or EOFError:
        default_predictions_dict = {}
    try:
        correct_predictions_dict = json.load(open(cache_correct_predictions_file, 'r'))
    except FileNotFoundError or EOFError:
        correct_predictions_dict = {}

    df = df.drop_duplicates(subset=['masked'], keep='first')
    for i, text in enumerate(tqdm(df['masked'], desc=f'Loading Default Predictions', leave=False)):
        if text in correct_predictions_dict:
            continue

        masked_results[text] = []
        for _, entry in names.iterrows():
            name = entry['name']
            gender = entry['gender']
            masked_text = text.replace('[NAME]', name).replace('[PRONOUN]', mask_token)
            if masked_text in default_predictions_dict:
                masked_results[text].append(default_predictions_dict[masked_text])
            else:
                predictions = evaluate_he_she(model, tokenizer, masked_text)
                predictions['gender'] = gender
                default_predictions_dict[masked_text] = predictions
                modified = True
                masked_results[text].append(predictions)

        he_probs = np.array([x['he'] for x in masked_results[text]])
        she_probs = np.array([x['she'] for x in masked_results[text]])
        genders = np.array([x['gender'] for x in masked_results[text]])
        most_likely_token = [x['most_likely_token'] for x in masked_results[text]]
        if len(np.shape(he_probs)) == 2:
            predicted_he = np.all(he_probs > she_probs, axis=-1)
            predicted_she = np.all(he_probs < she_probs, axis=-1)
        else:
            predicted_he = he_probs > she_probs
            predicted_she = he_probs < she_probs

        predicted_token_he = [all(x == 'he' for x in entry) if isinstance(entry, list) else entry == 'he' for entry in most_likely_token]
        predicted_token_she = [all(x == 'she' for x in entry) if isinstance(entry, list) else entry == 'she' for entry in most_likely_token]

        predicted_gender_correct = ((predicted_he & (genders == 'M') & predicted_token_he) | (predicted_she & (genders == 'F')) & predicted_token_she)

        # Calculate the proportion of correct predictions
        proportion_correct = predicted_gender_correct.mean()
        correct_predictions_dict[text] = proportion_correct

        if i > 0 and modified and i % 1000 == 0:
            print(f'Saved intermediate result with {len(default_predictions_dict)} entries to {cache_file}')
            pickle.dump(default_predictions_dict, open(cache_file, 'wb'))
            json.dump(correct_predictions_dict, open(cache_correct_predictions_file, 'w+'), indent=2)
            modified = False

    if modified:
        pickle.dump(default_predictions_dict, open(cache_file, 'wb'))


    df['filter'] = df['masked'].map(correct_predictions_dict)
    filtered_df = df[df['filter'] >= threshold].drop(columns=['filter']).reset_index(drop=True)
    print(f'Filtered {len(df)} to {len(filtered_df)} with threshold {threshold}')
    output = f'data/{df_name}_{model_name}_{threshold}.csv'
    filtered_df.to_csv(output, index=False)
    split(output)

    if model_name == 'bert-base-uncased':
        output = f'data/{df_name}_{threshold}.csv'
        filtered_df.to_csv(output, index=False)
        split(output)

def preprocess_text(text, sentences_per_snippet=2):
    # Step 1: Split text into paragraphs based on double newline with spaces or tabs in between
    paragraphs = re.split(r'\n\s*\n', text.strip())

    snippets = []

    # Step 2: Process each paragraph
    for paragraph in paragraphs:
        # Split the paragraph into sentences
        sentences = sent_tokenize(paragraph)

        # Step 3: Create snippets of the specified number of sentences
        i = 0
        while i < len(sentences):
            snippet = ' '.join(sentences[i:i + sentences_per_snippet]).replace('\n', ' ')
            snippets.append(snippet)
            i += sentences_per_snippet  # Move to the next group of sentences

    return snippets

def filter_text(text, num_proc=None, num_batches=1):
    num_proc = num_proc or os.cpu_count() - 1

    df, df_all_names = get_names_datasets()

    if isinstance(text, list):
        dataset = Dataset.from_dict({'text': text})
    elif isinstance(text, Dataset):
        dataset = text
    else:
        raise ValueError(f'Unknown type {type(text)}')
    filter = GenderTextFilter(df, df_all_names)
    filtered_dataset = dataset.map(filter, num_proc=num_proc, suffix_template="_{rank:05d}_of_{num_proc:05d}_of_" + str(num_batches))
    filtered_dataset = filtered_dataset.filter(lambda row: row['masked'])
    return filtered_dataset

def filter_book(book_path):
    text = open(book_path, 'r').read()
    text_snippets = preprocess_text(text)
    dataset = filter_text(text_snippets)
    dataset.to_csv(book_path.replace('.txt', '.csv'))


def filter_dataset(dataset_id, batch_size=10000, split='validation'):
    dataset = load_dataset(dataset_id, split=split)
    text = dataset['text']

    # Determine the number of batches
    num_batches = (len(text) + batch_size - 1) // batch_size

    # Process batches
    processed_datasets = []
    for i in tqdm(range(num_batches)):
        batch_file = f'data/cache/{dataset_id}_{split}/batch_{i}.csv'

        # If intermediate batch file exists, load it
        if os.path.exists(batch_file):
            batch_dataset = Dataset.from_csv(batch_file)
            print(f'Loaded batch {i} from {batch_file}')
        else:
            # Process the batch
            print(f'Processing batch {i} of {num_batches}')

            batch_text = text[i * batch_size: (i + 1) * batch_size]
            # Flatten the list of texts after preprocessing
            batch_text = [x for t in batch_text for x in preprocess_text(t)]
            batch_dataset = filter_text(batch_text, num_batches=num_batches)
            batch_dataset.to_csv(batch_file, index=False)
            print(f'Saved batch {i} to {batch_file} with {len(batch_dataset)} samples')

        processed_datasets.append(batch_dataset)

    # Concatenate all batches into one dataset
    full_dataset = concatenate_datasets(processed_datasets)

    # Save the full dataset
    full_dataset.to_csv(f'data/{dataset_id}_{split}.csv', index=False)

    df = full_dataset.to_pandas()
    df = df[~(df['gender_words'] | df['other_gender_pronouns'])].reset_index(drop=True)
    filter_by_model_confidence('bert-base-uncased', threshold=0.98, df=df, df_name=f'{dataset_id}_{split}')


def generate_geneutral():
    filter_no_gender_text()

def generate_genter():
    filter_gender_text()
    filter_by_model_confidence('bert-base-uncased', threshold=1.0)

# potential other sources for GENTER
# roneneldan/TinyStories
# ajibawa-2023/General-Stories-Collection
# ajibawa-2023/Children-Stories-Collection

if __name__ == '__main__':
    generate_geneutral()
    generate_genter()