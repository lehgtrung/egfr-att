
import pandas as pd
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def train_validation_split(data_path):
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
    return train_test_split(data, test_size=0.2)


class EGFRDataset(data.Dataset):
    def __init__(self, data):
        self.data = data
        self.NON_MORD_NAMES = ['smile_ft', 'id', 'subset', 'quinazoline', 'pyrimidine', 'smiles', 'active']

        # Standardize mord features
        scl = StandardScaler()
        self.mord_ft = scl.fit_transform(self.data.drop(columns=self.NON_MORD_NAMES)).tolist()
        self.non_mord_ft = self.data['smile_ft'].values.tolist()
        self.label = self.data['active'].values.tolist()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.mord_ft[idx], self.non_mord_ft[idx], self.label[idx]

    def get_dim(self, ft):
        if ft == 'non_mord':
            return len(self.non_mord_ft[0])
        if ft == 'mord':
            return len(self.mord_ft[0])







