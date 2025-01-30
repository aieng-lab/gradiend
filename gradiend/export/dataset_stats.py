from gradiend.data import read_namexact, read_namextend, read_genter, read_geneutral, read_gentypes


def print_dataset_statistics():

    namexact = read_namexact
    namextend = read_namextend
    genter = read_genter
    geneutral = read_geneutral
    gentypes = read_gentypes

    datasets = {
        "namexact": namexact,
        "namextend": namextend,
        "genter": genter,
        "geneutral": geneutral,
        "gentypes": gentypes,
    }

    for name, dataset in datasets.items():

        fulL_dataset = dataset()
        print(f"Dataset: {name}")
        print(f"Number of entries: {len(fulL_dataset)}")
        # number of entries per split if dataset accepts split argument
        try:
            for split in ['train', 'val', 'test']:
                split_dataset = dataset(split=split)
                print(f"Number of entries in {split} split: {len(split_dataset)}")
        except TypeError:
            pass
        print('\n')

if __name__ == '__main__':
    print_dataset_statistics()