import numpy as np

def make_diagonal(x):
    # Your code here
    n = len(x)
    c = np.zeros((n, n), dtype=float)
    x = np.asarray(x, dtype=float)
    np.fill_diagonal(c, x)
    return c