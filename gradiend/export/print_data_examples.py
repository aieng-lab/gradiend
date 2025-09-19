import pandas as pd

ds = 'data/race/asian_to_black'
ds = 'data/religion/christian_to_jewish'
from datasets import load_dataset, Dataset

ds = Dataset.load_from_disk(ds)

# print 10 smallest examples wrt 'text' length
# Sort the dataset by the length of 'text'
sorted_ds = sorted(ds, key=lambda x: len(x['text']))

# Print the 10 smallest examples
#for example in sorted_ds[500: 600]:
#    print(example)



# print ds sizes

main_folders = ['data/race', 'data/religion']

# print sizes of datasets in each main folder
for main_folder in main_folders:
    min_size = float('inf')
    max_size = 0

    print(f"Sizes of datasets in {main_folder}:")
    import os
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            ds = Dataset.load_from_disk(subfolder_path)
            n = len(set(ds['text']))

            print(f"  {subfolder}: {len(ds)} examples ({n} unique texts)")
            if n < min_size:
                min_size = n
            if n > max_size:
                max_size = n
    print(f"  Min size: {min_size}")
    print(f"  Max size: {max_size}")


df = pd.read_csv('data/gerneutral.csv')
print(f"Geneutral dataset size: {len(df)} examples")
unique_texts = df['text'].nunique()
print(f"Geneutral dataset unique texts: {unique_texts}")