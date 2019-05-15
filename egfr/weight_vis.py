
import torch
import argparse
import pandas as pd
from nets import UnitedNet
from dataset import EGFRDataset
from torch.utils.data import dataloader
import utils
from rdkit import Chem
from rdkit.Chem.Draw import SimilarityMaps
from feature import atom_flag
import os
import numpy as np
import ast


def get_mol_importance(data_path, model_path, dir_path, device):
    data = pd.read_json(data_path, lines=True)
    dataset = EGFRDataset(data, infer=True)
    loader = dataloader.DataLoader(dataset=dataset,
                                   batch_size=128,
                                   collate_fn=utils.custom_collate,
                                   shuffle=True)
    united_net = UnitedNet(dense_dim=dataset.get_dim('mord'), use_mat=True, dir_path=dir_path, infer=True).to(device)
    united_net.load_state_dict(torch.load(model_path, map_location=device))
    united_net.eval()

    for i, (smiles, mord_ft, non_mord_ft, label) in enumerate(loader):
        with torch.no_grad():
            mord_ft = mord_ft.float().to(device)
            non_mord_ft = non_mord_ft.view((-1, 1, 150, 42)).float().to(device)
            mat_ft = non_mord_ft.squeeze(1).float().to(device)
            # Forward to get smiles and equivalent weights
            o = united_net(non_mord_ft, mord_ft, mat_ft, smiles=smiles)
    print('Forward done !!!')


def weight_vis(smiles, weights, cm='jet', lines=10):
    m = Chem.MolFromSmiles(smiles)
    try: smi = Chem.MolToSmiles(m, kekuleSmiles=True, isomericSmiles=True, rootedAtAtom=int(n))
    except: smi = Chem.MolToSmiles(m, kekuleSmiles=True, isomericSmiles=True)
    smi = Chem.MolToSmiles(m)
    aod = ast.literal_eval(m.GetProp('_smilesAtomOutputOrder'))
    flg = atom_flag(smi,150)
    exwt = [weights[i] for i in range(len(weights)) if flg[i]]
    fig = SimilarityMaps.GetSimilarityMapFromWeights(m, exwt, colorMap=cm, contourLines=lines)
    return fig


def save_weight_vis(smiles_file, weight_file, save_dir):
    with open(smiles_file) as f:
        smiles = f.read().splitlines()

    weights = np.loadtxt(weight_file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(len(smiles)):
        filename = os.path.join(save_dir, str(i) + '.png')
        fig = weight_vis(smiles[i], weights[i])
        fig.savefig(filename, bbox_inches="tight", pad_inches=0)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Input dataset', dest='dataset', default='data/egfr_10_full_ft_pd_lines.json')
    parser.add_argument('-m', '--modelpath', help='Input dataset', dest='modelpath', default='data/model')
    parser.add_argument('-dir', '--dirpath', help='Directory to save attention weights', dest='dirpath', default='data/att_weight')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    get_mol_importance(args.dataset, args.modelpath, args.dirpath, args.device)
    save_weight_vis(os.path.join(args.dirpath,'smiles.txt'),
                    os.path.join(args.dirpath, 'weight.txt'),
                    os.path.join(args.dirpath, 'vis/')
                    )


if __name__ == '__main__':
    main()