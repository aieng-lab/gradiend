import pandas as pd


def split(file, data=None, prop_val=0.025, prop_test=0.1):
    # Load the data from a CSV file if not already provided
    if data is None:
        data = pd.read_csv(file)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split the data into training and testing sets

    n_val = int(data.shape[0] * prop_val)
    n_test = int(data.shape[0] * prop_test)

    val_data = data[:n_val].copy()
    test_data = data[n_val: n_val + n_test].copy()
    train_data = data[n_val + n_test:].copy()

    print(f"Training set: {train_data.shape[0]} samples")
    print(f"Validation set: {val_data.shape[0]} samples")
    print(f"Test set: {test_data.shape[0]} samples")

    output_prefix = file.removesuffix('.csv')
    train_data.to_csv(output_prefix + '_train.csv', index=False)
    val_data.to_csv(output_prefix + '_val.csv', index=False)
    test_data.to_csv(output_prefix + '_test.csv', index=False)

    train_data['split'] = 'train'
    val_data['split'] = 'val'
    test_data['split'] = 'test'
    combined = pd.concat([train_data, val_data, test_data])
    combined.to_csv(output_prefix + '_split.csv', index=False)