"Data normalization for the training database"
import pickle
from os import path

import pandas as pd

NORMALIZED_CONSTANTS_FILE = path.join(path.dirname(__file__), "udata/normalization-consts.pkl")

def normalize_training_data(input_db_path):
    """Normalizes data with respect to mean and std-dev, stores both variables
    for later inference.
    """
    
    dataset = pd.read_json(input_db_path)

    dataset = dataset.loc[dataset.loc[:, 'sub_harmonic'] != 0.0] #quitando los valores cuando subarmonico dif a cero 

    desc = dataset.describe()
    desc = desc.transpose()

    normalize = lambda x:(x - desc["mean"]/desc["std"])
    normalized_data = normalize(dataset)

    with open(NORMALIZED_CONSTANTS_FILE, "wb") as f:
        serialized = pickle.dumps((desc["mean"], desc["std"]))
        f.write(serialized)

    return normalized_data

class Normalizer:
    """utility class to normalize inference inputs using the same mean and std-dev
    as those used when training the algorithms"""

    def __init__(self):
        with open(NORMALIZED_CONSTANTS_FILE, "rb") as f:
            serialized = f.read()
        (mean, std) = pickle.loads(serialized)
        self.mean = mean
        self.std = std
        
    def normalize_row(self, r):
        return (r - self.mean)/self.std
