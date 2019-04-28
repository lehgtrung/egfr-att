
import torch
import torch.nn as nn
import utils
import pandas as pd
from nets import UnitedNet
from dataset import EGFRDataset


def get_mol_importance(data_path, model_path, device):
    data = pd.read_json(data_path, lines=True)
    dataset = EGFRDataset(data)
    smile_ft = dataset.get_smile_ft()

    united_net = UnitedNet(dense_dim=dataset.get_dim('mord'), use_mat=True).to(device)
    united_net.load_state_dict(torch.load(model_path, map_location=device))
    comb_weight = united_net.state_dict()['att_fc.weight']
    comb_bias = united_net.state_dict()['att_fc.bias']






