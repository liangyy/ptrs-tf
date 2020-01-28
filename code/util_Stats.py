import scipy.stats
import numpy as np
def inv_norm_col(mat):
    return np.apply_along_axis(inv_norm_vec, 0, mat)
def inv_norm_vec(vec, offset = 1):
    rank = myrank(vec)
    return scipy.stats.norm.ppf(rank / (len(rank) + offset), loc = 0, scale = 1)
def myrank(vec):
    argsort = np.argsort(vec)
    ranks = np.empty_like(argsort)
    ranks[argsort] = np.arange(len(vec))
    return ranks + 1  # rank starts from 1

