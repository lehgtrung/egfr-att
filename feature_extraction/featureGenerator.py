# -*- coding: utf-8 -*-

from rdkit import Chem
from feature import mol_to_feature
import numpy as np
import pandas as pd
import pickle

MAX_LEN = 150
INPUT_SMILES = '../data/egfr.csv'
OUTPUT = '../data/data.pickle'
SMILE = 'smiles'
ACTIVE = 'active'


if __name__ == "__main__":
    ft = []
    df = pd.read_csv(INPUT_SMILES)
    for i in range(len(df)):
        try:
            mol = Chem.MolFromSmiles(df[SMILE][i])
            ft.append(mol_to_feature(mol,-1, 150))
        except:
            print(i)
    
    dt = {'feature': np.array(ft),
          'target':df[ACTIVE].values}
    
    with open(OUTPUT, 'wb') as handle:
        pickle.dump(dt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    