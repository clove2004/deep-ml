import numpy as np

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> list:
    """
    L2-normalize x along the given axis.

    Args:
        x: input NumPy array
        axis: axis along which to normalize
        eps: small constant for numerical stability

    Returns:
        Normalized array as a nested Python list.
    """
    # Your code here
    if x.ndim == 1:
        su = 0
        for i in x:
            su += i**2
        su = np.sqrt(su)
        if su != 0:
            x = x/su
        else:
            x = np.zeros_like(x, dtype=float)
        return x.tolist()
    ans = []
    m, n = x.shape
    if axis == 0:
        for j in range (n):
            su = 0
            for i in range (m):
                su += x[i][j]**2
            su = np.sqrt(su)
            if su != 0:
                x[:, j] = x[:, j]/su 
            else:
                x[:, j] = np.zeros_like(x[:, j], dtype=float)
    else:
        for i in range (m):
            su = 0
            for j in range (n):
                su += x[i][j]**2
            su = np.sqrt(su)
            if su != 0:
                x[i, :] = x[i, :]/su 
            else:
                x[i, :] = np.zeros_like(x[i, :], dtype=float)
    return x.tolist()
    pass
