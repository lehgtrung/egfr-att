
import pandas as pd
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os, glob


def read_data(data_path):
    data = None
    if data_path.endswith('.json'):
        try:
            data = pd.read_json(data_path, lines=True)
        except ValueError:
            data = pd.read_json(data_path)
    if data_path.endswith('.zip'):
        try:
            data = pd.read_json(data_path, compression='zip', lines=True)
        except ValueError:
            data = pd.read_json(data_path, compression='zip')
    return data


def train_validation_split(data_path):
    train_path = data_path.split('.')[0] + '_' + 'train.json'
    val_path = data_path.split('.')[0] + '_' + 'val.json'
    if os.path.exists(train_path) and os.path.exists(val_path):
        return read_data(train_path), read_data(val_path)
    data = read_data(data_path)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data.to_json(train_path, orient='records', lines=True)
    val_data.to_json(val_path, orient='records', lines=True)
    return train_data, val_data


def train_cross_validation_split(data_path):
    dir_path = os.path.dirname(os.path.abspath(data_path))
    fold_dirs = glob.glob(os.path.join(dir_path, 'folds_*'))
    if len(fold_dirs) > 0:
        for fold_dir in fold_dirs:
            train_path = os.path.join(fold_dir, 'train.json')
            val_path = os.path.join(fold_dir, 'val.json')
            yield pd.read_json(train_path), pd.read_json(val_path)
    else:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        data = read_data(data_path)
        for i, (train_ids, val_id) in enumerate(kfold.split(X=np.empty((len(data['active']), 1)),
                                                            y=data['active'])):
            train_data = data.iloc[train_ids, ]
            val_data = data.iloc[val_id, ]
            os.makedirs(os.path.join(dir_path, 'folds_{}'.format(i)), exist_ok=True)
            train_data.to_json(os.path.join(os.path.join(dir_path, 'folds_{}'.format(i)), 'train.json'))
            val_data.to_json(os.path.join(os.path.join(dir_path, 'folds_{}'.format(i)), 'val.json'))

            yield data.iloc[train_ids, ], data.iloc[val_id, ]


class EGFRDataset(data.Dataset):
    def __init__(self, data, infer=False):
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = read_data(data)
        self.NON_MORD_NAMES = ['smile_ft', 'id', 'subset', 'quinazoline', 'pyrimidine', 'smiles', 'active']
        self.infer = infer

        # Standardize mord features
        scl = StandardScaler()
        self.mord_ft = scl.fit_transform(
            self.data.drop(columns=self.NON_MORD_NAMES).astype(np.float64)).tolist()
        self.non_mord_ft = self.data['smile_ft'].values.tolist()
        self.smiles = self.data['smiles'].values.tolist()
        self.label = self.data['active'].values.tolist()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if self.infer:
            return self.smiles[idx], self.mord_ft[idx], self.non_mord_ft[idx], self.label[idx]
        else:
            return self.mord_ft[idx], self.non_mord_ft[idx], self.label[idx]

    def get_dim(self, ft):
        if ft == 'non_mord':
            return len(self.non_mord_ft[0])
        if ft == 'mord':
            return len(self.mord_ft[0])

    def get_smile_ft(self):
        return self.non_mord_ft







