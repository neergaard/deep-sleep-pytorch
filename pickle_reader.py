import pickle

import numpy as np


def read_pickle(filepath):
    '''Read contents of the output pickle files.
    
    Usage:
    from pickle_reader import read_pickle
    
    y, t = read_pickle(filename)
    
    Args:
        filepath: str -- path to pickle file
        
    Returns:
        y: ndarray -- output predictions every second
        t: ndarray -- targets every second (hypnogram labels)
    '''
    # This loads contents of pickle file into a dict
    with open(filepath, 'rb') as pkl:
        out = pickle.load(pkl)

    # Isolate and reshape targets from (N, 300) into (N * 300,)
    targets = np.asarray(out['targets']).reshape(-1)

    # Isolate and reshape predictions from (N, 5, 300) into (N * 300, 5)
    predictions = np.asarray(out['predictions'])
    N, K, T = predictions.shape
    predictions = predictions.transpose(1, 0, 2).reshape(K, N * T).T

    return predictions, targets
