import os

from datasets import Dataset

from gradiend.data import read_genter, read_namexact, read_namextend, read_geneutral, read_gentypes

org_id = 'aieng-lab'
token = os.getenv('HF_TOKEN')

splits = {
    'train': 'train',
    'val': 'validation',
    'test': 'test',
}

def upload_genter():
    for split, hf_split in splits.items():
        df = read_genter(split=split)
        df = df[['text', 'masked', 'label', 'name', 'pronoun', 'pronoun_count']]
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(f'{org_id}/genter', split=hf_split, token=token)

def upload_namexact():
    name_exact_columns = ['name', 'gender', 'count', 'probability', 'split']

    df = read_namexact()
    df = df[name_exact_columns]
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(f'{org_id}/namexact', split='all', token=token, set_default=True)

    for split, hf_split in splits.items():
        df = read_namexact(split=split)
        df = df[name_exact_columns]
        dataset = Dataset.from_pandas(df)
        dataset.push_to_hub(f'{org_id}/namexact', split=hf_split, token=token, set_default=False)

def upload_namextend():
    # gender_agreement in [0.5, 1.0]
    # genders M, F,
    df = read_namextend()
    df = df[['name', 'gender', 'count', 'probability', 'gender_agreement', 'prob_F', 'prob_M', 'primary_gender', 'genders']]
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(f'{org_id}/namextend', token=token)

def upload_geneutral():
    # geneutral
    df = read_geneutral()
    df = df[['text']]
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(f'{org_id}/geneutral', token=token)

def upload_gentypes():
    # gentypes
    df = read_gentypes()
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(f'{org_id}/gentypes', token=token)


upload_gentypes()
upload_geneutral()
upload_genter()
upload_namexact()
upload_namextend()
