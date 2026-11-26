import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Perform QR decomposition using Gram-Schmidt process.
    
    Args:
        A: An m x n matrix represented as list of lists
    
    Returns:
        Tuple of (Q, R) where Q is orthogonal and R is upper triangular
    """
    # Your code here
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = []
    v = []
    a = []
    for j in range (n):
        vj = A[:,j].copy()
        aj = A[:,j].copy()
        for i in range (0, j):
            vj -= (Q[i] @ aj) * Q[i]
        qj = vj/np.linalg.norm(vj, ord=2)
        Q.append(qj)
        v.append(vj)
        a.append(aj)
    R = np.zeros((n, n), dtype=float)
    for i in range (n):
        for j in range (n):
            if i < j:
                R[i][j] = Q[i]@a[j]
            elif i == j:
                R[i][j] = np.linalg.norm(v[j], ord=2)
            else:
                R[i][j] = 0
    Q = np.asarray(Q)
    R = np.asarray(R)
    Q = np.transpose(Q)
    return (Q.tolist(), R.tolist())
    pass