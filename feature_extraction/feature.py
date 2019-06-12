# Credit: http://www.dna.bio.keio.ac.jp/smiles/

from rdkit import Chem
from rdkit.Chem import rdchem
import re

Chiral = {"CHI_UNSPECIFIED": 0,
          "CHI_TETRAHEDRAL_CW": 1,
          "CHI_TETRAHEDRAL_CCW": 2,
          "CHI_OTHER": 3}
Hybridization = {"UNSPECIFIED": 0,
                 "S": 1,
                 "SP": 2,
                 "SP2": 3,
                 "SP3": 4,
                 "SP3D": 5,
                 "SP3D2": 6,
                 "OTHER": 7}

atomInfo = 21
structInfo = 21
lensize = atomInfo + structInfo

H_Vector = [0] * atomInfo
H_Vector[0] = 1

lowerReg = re.compile(r'^[a-z]+$')
upperReg = re.compile(r'^[A-Z]+$')


def islower(s):
    return lowerReg.match(s) is not None


def isupper(s):
    return upperReg.match(s) is not None


def calc_atom_feature(atom):
    if atom.GetSymbol() == 'H':
        feature = [1, 0, 0, 0, 0]
    elif atom.GetSymbol() == 'C':
        feature = [0, 1, 0, 0, 0]
    elif atom.GetSymbol() == 'O':
        feature = [0, 0, 1, 0, 0]
    elif atom.GetSymbol() == 'N':
        feature = [0, 0, 0, 1, 0]
    else:
        feature = [0, 0, 0, 0, 1]

    feature.append(atom.GetTotalNumHs() / 8)
    feature.append(atom.GetTotalDegree() / 4)
    feature.append(atom.GetFormalCharge() / 8)
    feature.append(atom.GetTotalValence() / 8)
    feature.append(atom.IsInRing() * 1)
    feature.append(atom.GetIsAromatic() * 1)

    f = [0] * (len(Chiral) - 1)
    if Chiral.get(str(atom.GetChiralTag()), 0) != 0:
        f[Chiral.get(str(atom.GetChiralTag()), 0)] = 1
    feature.extend(f)

    f = [0] * (len(Hybridization) - 1)
    if Hybridization.get(str(atom.GetHybridization()), 0) != 0:
        f[Hybridization.get(str(atom.GetHybridization()), 0)] = 1
    feature.extend(f)

    return feature


def calc_structure_feature(c, flag, label):
    feature = [0] * structInfo

    if c == '(':
        feature[0] = 1
        flag = 0
    elif c == ')':
        feature[1] = 1
        flag = 0
    elif c == '[':
        feature[2] = 1
        flag = 0
    elif c == ']':
        feature[3] = 1
        flag = 0
    elif c == '.':
        feature[4] = 1
        flag = 0
    elif c == ':':
        feature[5] = 1
        flag = 0
    elif c == '=':
        feature[6] = 1
        flag = 0
    elif c == '#':
        feature[7] = 1
        flag = 0
    elif c == '\\':
        feature[8] = 1
        flag = 0
    elif c == '/':
        feature[9] = 1
        flag = 0
    elif c == '@':
        feature[10] = 1
        flag = 0
    elif c == '+':
        feature[11] = 1
        flag = 1
    elif c == '-':
        feature[12] = 1
        flag = 1
    elif c.isdigit():
        if flag == 0:
            if c in label:
                feature[20] = 1
            else:
                label.append(c)
                feature[19] = 1
        else:
            feature[int(c) - 1 + 12] = 1
            flag = 0
    return feature, flag, label


def calc_featurevector(mol, smiles, atomsize):
    flag = 0
    label = []
    molfeature = []
    idx = 0
    j = 0

    for c in smiles:
        if islower(c):
            continue
        elif isupper(c):
            if c == 'H':
                molfeature.extend(H_Vector)
            else:
                molfeature.extend(calc_atom_feature(rdchem.Mol.GetAtomWithIdx(mol, idx)))
                idx = idx + 1
            molfeature.extend([0] * structInfo)
            j = j + 1

        else:
            molfeature.extend([0] * atomInfo)
            f, flag, label = calc_structure_feature(c, flag, label)
            molfeature.extend(f)
            j = j + 1

    # 0-Padding
    molfeature.extend([0] * (atomsize - j) * lensize)
    return molfeature


def mol_to_feature(mol, n, atomsize):
    try:
        defaultSMILES = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=True, rootedAtAtom=int(n))
    except ValueError:
        defaultSMILES = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=True)
    try:
        isomerSMILES = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True, rootedAtAtom=int(n))
    except ValueError:
        isomerSMILES = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)
    return calc_featurevector(Chem.MolFromSmiles(defaultSMILES), isomerSMILES, atomsize)


def mol_to_allSMILESfeature(mol, atomsize):
    idx, features = 0, []
    while idx < mol.GetNumAtoms():
        try:
            defaultSMILES = Chem.MolToSmiles(mol, kekuleSmiles=False, isomericSmiles=True, rootedAtAtom=int(idx))
        except ValueError:
            break
        isomerSMILES = Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True, rootedAtAtom=int(idx))
        features.append(calc_featurevector(Chem.MolFromSmiles(defaultSMILES), isomerSMILES, atomsize))
        idx = idx + 1
    return features
