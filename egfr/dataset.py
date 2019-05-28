
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
    return train_test_split(data, test_size=0.2, random_state=42)


def train_cross_validation_split(data_path):
    fold_dirs = glob.glob(os.path.join(data_path, 'folds_*'))
    if len(fold_dirs) > 0:
        for fold_dir in fold_dirs:
            train_path = os.path.join(fold_dir, 'train.json')
            val_path = os.path.join(fold_dir, 'val.json')
            yield read_data(train_path), read_data(val_path)
    else:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        data = read_data(data_path)
        for train_ids, val_id in kfold.split(X=np.empty((len(data['active']), 1)), y=data['active']):
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
        self.mord_ft = scl.fit_transform(self.data.drop(columns=self.NON_MORD_NAMES)).tolist()
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

    def persist(self, mode):
        if not os.path.exists('data/egfr_10_full_ft_pd_lines_{}.json'.format(mode)):
            self.data.to_json('data/egfr_10_full_ft_pd_lines_{}.json'.format(mode),
                              orient='records', lines=True)







