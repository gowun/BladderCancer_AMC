import pandas as pd 
import numpy as np 
import pickle

def save_data(obj, path, mode):
  if mode == 'csv':
    obj.to_csv(path, index=False)
  elif mode == 'pickle':
    pickle.dump(obj, open(path, 'wb'), 4)


def load_data(path, mode):
    if mode == 'csv':
        return pd.read_csv(path)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))