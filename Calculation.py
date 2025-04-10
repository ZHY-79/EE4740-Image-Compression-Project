import numpy as np


def compute_metrics(x_original, x_recovered):

    x_original = np.array(x_original)
    x_recovered = np.array(x_recovered)
    mse = np.linalg.norm(x_original - x_recovered) ** 2
    x_original_norm = x_original / np.linalg.norm(x_original)
    x_recovered_norm = x_recovered / np.linalg.norm(x_recovered)
    nmse = np.linalg.norm(x_original_norm - x_recovered_norm) ** 2
    
    return mse, nmse