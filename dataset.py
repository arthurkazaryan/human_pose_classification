import pandas as pd
import h5py
import tqdm
import numpy as np
from ast import literal_eval
from pathlib import Path
from tensorflow.keras.utils import to_categorical


dataframe = pd.read_csv(Path.cwd().joinpath('dataset', 'dataframe_coords.csv'), delimiter=';')

# Getting classes names ['front', 'side', 'back']
classes_names = dataframe.loc[:, 'positions'].unique().tolist()[:-1]
count_each_class = dataframe.loc[:, 'positions'].value_counts().to_list()[:-1]
classes_dict = {classes_names[i]: count_each_class[i] for i in range(len(classes_names))}
split = {'train': 0.80, 'val': 0.1995, 'test': 0.0005}
split_data = ['front', 'side', 'back']

front_pos = dataframe.loc[dataframe['positions'] == 'front']
side_pos = dataframe.loc[dataframe['positions'] == 'side']
back_pos = dataframe.loc[dataframe['positions'] == 'back']

# Since the amount of images for each classes are different, I am taking the same percentage from each class.
train_data = []
for split_name in split_data:
    train_data.append(globals()[f'{split_name}_pos'][:int(split['train']*classes_dict[split_name])])
pd.concat(train_data).sample(frac=1).reset_index(drop=True).to_csv(Path.cwd().joinpath('dataset', 'train.csv'))
val_data = []
for split_name in split_data:
    val_data.append(globals()[f'{split_name}_pos'][:int(split['val']*classes_dict[split_name])])
pd.concat(val_data).sample(frac=1).reset_index(drop=True).to_csv('val.csv')

train_dataframe = pd.read_csv('train.csv', index_col=0)
val_dataframe = pd.read_csv('val.csv', index_col=0)

with h5py.File(Path.cwd().joinpath('dataset', 'dataset.h5'), 'w') as hdf:
    for split in ['train', 'val']:
        hdf.create_group(split)
        hdf[split].create_group('input')
        cur_df = globals()[f'{split}_dataframe']
        for i in tqdm.tqdm(range(len(cur_df)), desc=f'{split} images'):
            coords_array = np.array([literal_eval(col) for col in cur_df.iloc[i, 2:].to_list()])
            hdf[f'{split}/input'].create_dataset(str(i), data=coords_array)
        hdf[split].create_group('output')
        for i in tqdm.tqdm(range(len(cur_df)), desc=f'{split} classes'):
            class_name = cur_df.iloc[i, 1]
            class_array = to_categorical(classes_names.index(class_name), 3, dtype='int8')
            hdf[f'{split}/output'].create_dataset(str(i), data=class_array)
