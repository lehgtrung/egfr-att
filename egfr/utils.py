
import os
import pickle


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)