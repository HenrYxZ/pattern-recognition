__author__ = 'lucas'

import numpy as np

def random_rows(matrix, sample_size):
    n_rows = matrix.shape[0]
    indexes = np.random.choice(n_rows, sample_size)
    return matrix[indexes,:]

