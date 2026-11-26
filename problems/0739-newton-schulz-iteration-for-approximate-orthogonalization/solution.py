import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    # Your code here
    M = np.asarray(M)
    if np.linalg.norm(M, ord='fro') == 0:
        m, n = M.shape
        return np.zeros((m, n), dtype=float)
    else:
        M = M/np.linalg.norm(M, ord='fro')
    for _ in range (num_iters):
        M1 = (M @ np.transpose(M))
        M = a*M + b* M1 @ M + c*(M1 @ M1) @ M
    return M
    pass
