import pandas as pd


def read_gerneutral(max_size=None, exclude=None):
    df = pd.read_csv("data/gerneutral.csv")

    if exclude:
        df = df[~df['text'].str.contains('|'.join(exclude), case=False, na=False)]

    if max_size is not None:
        df = df.head(min(max_size, len(df)))
    return df