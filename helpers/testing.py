import torch
import torch.nn as nn
import numpy as np

#
# Helper
#
def generate_data_points(a, b, n_samples, noise_var):
    """
    Generate data points with noise.
    
    Parameters:
    a (float): The start of the range.
    b (float): The end of the range.
    n_samples (int): Number of samples to generate.
    noise_var (float): Variance of the Gaussian noise.
    
    Returns:
    tuple: X and y values reshaped as columns.
    """
    X = np.linspace(a, b, n_samples)
    y = 2**np.cos(X**2) + np.random.normal(0, noise_var, n_samples)

    return X.reshape(-1, 1), y.reshape(-1, 1)

#
# 
#