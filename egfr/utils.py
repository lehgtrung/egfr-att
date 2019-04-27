
import os
import pickle
import torch


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_model(model, model_dir_path, hash_code, e):
    """
    :param model: training model
    :param model_dir_path: directory path
    :param hashcode: hashcode
    :param e: epoch
    """
    torch.save(model.state_dict(), "{}/model_{}_{}".format(model_dir_path, hash_code, e + 1))


def save_checkpoint():
    pass


