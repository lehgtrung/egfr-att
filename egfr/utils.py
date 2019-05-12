
import os
import pickle
import torch
import collections


def get_max_length(x):
    return len(max(x, key=len))


def pad_sequence(seq):
    def _pad(_it, _max_len):
        return [0] * (_max_len - len(_it)) + _it
    padded = [_pad(it, get_max_length(seq)) for it in seq]
    return padded


def custom_collate(batch):
    """
        Custom collate function for our batch, a batch in dataloader looks like
            [(0, [24104, 27359], 6684),
            (0, [24104], 27359),
            (1, [16742, 31529], 31485),
            (1, [16742], 31529),
            (2, [6579, 19316, 13091, 7181, 6579, 19316], 13091)]
    """
    transposed = zip(*batch)
    lst = []
    for samples in transposed:
        if isinstance(samples[0], str):
            lst.append(samples)
        if isinstance(samples[0], int):
            lst.append(torch.LongTensor(samples))
        elif isinstance(samples[0], float):
            lst.append(torch.DoubleTensor(samples))
        elif isinstance(samples[0], list):
            lst.append(torch.LongTensor(pad_sequence(samples)))
    return lst


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(model, model_dir_path, hash_code):
    """
    :param model: training model
    :param model_dir_path: directory path
    :param hash_code: hashcode
    :param e: epoch
    """
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    torch.save(model.state_dict(), "{}/model_{}_{}".format(model_dir_path, hash_code, "BEST"))
    print('Save done!')

def load_model():
    pass

