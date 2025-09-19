import os

import pandas as pd
from gradiend.data.split import split as data_split


# note that the splits are no split subsets for different options!
def read_processed_bookcorpus(filter_gender_words=True, split=None, threshold=1.0):
    if threshold is None:
        prefix = 'data/bookcorpus'
    else:
        prefix = f'data/genter_{threshold}'

    base_file = f'{prefix}.csv'
    if not os.path.isfile(base_file):
        if threshold is None:
            raise ValueError(f'GENTER base data is not available: {base_file}')

        print(f'GENTER data is not filtered yet for threshold {threshold}! Running data_filtering.filter_by_model_confidence first!')
        from filtering import filter_by_model_confidence
        filter_by_model_confidence(threshold=threshold)
        assert os.path.isfile(base_file)

    split_suffix = f'_{split}' if split else '_split'
    filter_gender_words_suffix = '' if filter_gender_words else '_no_filtered_gender_words'
    file = f'data/genter_{threshold}{split_suffix}{filter_gender_words_suffix}.csv'

    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        df = pd.read_csv(base_file)
        if filter_gender_words:
            # filter the gender words already here to get a proper train test split
            df = df[~(df['gender_words'] | df['other_gender_pronouns'])].reset_index(drop=True)

        if threshold is None:
            return df

        print(f'Applying split for {base_file}')
        data_split(base_file, data=df)
        df = pd.read_csv(file)

    return df