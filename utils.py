import numpy as np

def load_data():
    pass

def Normalize(x: np.ndarray):
    """
    Scale to 0-1
    """
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

def Standardize(x: np.ndarray):
    """
    Scale to zero mean unit variance
    """
    return (x - x.mean(axis=0)) / np.std(x, axis=0)