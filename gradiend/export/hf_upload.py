import os

import pandas as pd
from datasets import Dataset

from gradiend.data import read_names_data, read_processed_bookcorpus, read_bookcorpus

org_id = 'aieng-lab'
token = os.getenv('HF_TOKEN')

splits = {
    'train': 'train',
    'val': 'validation',
    'test': 'test',
}

def upload_genter():
    # as we don't have the permission to publish a subset of BookCorpus on Hugging Face, we need to map the text
    # to indices corresponding to the original dataset
    bookcorpus = read_bookcorpus()
    # Map the text in geneutral to indices in the original BookCorpus
    bookcorpus_mapping = {text: idx for idx, text in enumerate(bookcorpus['text'])}

    for split, hf_split in splits.items():
        df = read_processed_bookcorpus(split=split, filter_gender_words=True, threshold=1.0)
        df = df[['text', 'masked', 'label', 'name', 'pronoun', 'pronoun_count']]

        # Add the mapped indices to the geneutral DataFrame
        df['index'] = df['text'].map(bookcorpus_mapping)

        # Check if any text couldn't be mapped
        unmapped = df[df['index'].isna()]
        if not unmapped.empty:
            print(f"Warning: {len(unmapped)} texts could not be mapped to the BookCorpus dataset.")

        # Drop rows with unmapped indices if necessary

        df = df.dropna(subset=['index'])

        df[['index', 'name', 'pronoun']].to_csv(f'data/genter_indices_{split}.csv', index=False)
        df.to_csv('data/genter_with_indices.csv', index=False)


        #dataset = Dataset.from_pandas(df)
        #dataset.push_to_hub(f'{org_id}/genter', split=hf_split, token=token)

def _read_namexact(split=None):
    return read_names_data(split=split)

def upload_namexact():
    name_exact_columns = ['name', 'gender', 'count', 'probability', 'split']

    for split, hf_split in splits.items():
        df = _read_namexact()
        df = df[name_exact_columns]
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(f'{org_id}/namexact', split=hf_split, token=token)

def upload_namextend():
    # gender_agreement in [0.5, 1.0]
    # genders M, F,
    df = read_names_data(split=None,
                           subset=_read_namexact,  # pass unambiguous names to split wrt to the subset split!
                           filter_excluded_words=False,
                           filter_non_unique=False,
                           minimum_count=100,
                           max_entries=None)()
    df = df[['name', 'gender', 'count', 'probability', 'gender_agreement', 'prob_F', 'prob_M', 'primary_gender', 'genders']]
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(f'{org_id}/namextend', token=token)

def upload_geneutral():
    # geneutral
    df = pd.read_csv('data/geneutral.csv')
    df = df[['text']]

    # as we don't have the permission to publish a subset of BookCorpus on Hugging Face, we need to map the text
    # to indices corresponding to the original dataset
    bookcorpus = read_bookcorpus()
    # Map the text in geneutral to indices in the original BookCorpus
    bookcorpus_mapping = {text: idx for idx, text in enumerate(bookcorpus['text'])}

    # Add the mapped indices to the geneutral DataFrame
    df['index'] = df['text'].map(bookcorpus_mapping)

    # Check if any text couldn't be mapped
    unmapped = df[df['index'].isna()]
    if not unmapped.empty:
        print(f"Warning: {len(unmapped)} texts could not be mapped to the BookCorpus dataset.")

    # Drop rows with unmapped indices if necessary

    df = df.dropna(subset=['index'])

    df[['index']].to_csv('data/geneutral_indices.csv', index=False)
    df.to_csv('data/geneutral_with_indices.csv', index=False)

    #dataset = Dataset.from_pandas(df)
    #dataset.push_to_hub(f'{org_id}/geneutral', token=token)

def upload_gentypes():
    # gentypes
    df = pd.read_csv('data/gentypes.csv')
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(f'{org_id}/gentypes', token=token)


#upload_gentypes()
#upload_geneutral()
#upload_genter()
#upload_namexact()
#upload_namextend()
