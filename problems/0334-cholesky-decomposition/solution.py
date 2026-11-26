import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # Your code here
    A = np.asarray(A)
    m, n = len(A), len(A[0])
    if m != n:
        return -1
    symetric = np.allclose(A, A.T)
    tr, vec = np.linalg.eig(A)
    ok = symetric and np.all(tr.real > 1e-10) and np.allclose(tr.imag, 0)
    if ok == False:
        return -1
    L = np.zeros((n, n), dtype=float)
    for i in range (n):
        for j in range (n):
            if i == j:
                L[i][j] = A[i][i]
                for k in range (i):
                    L[i][j] -= L[i][k]**2
                L[i][j] = np.sqrt(L[i][j])
            else:
                L[i][j] = A[i][j]
                for k in range (j):
                    L[i][j] -= L[i][k]*L[j][k]
                if L[j][j] != 0:
                    L[i][j] /= L[j][j]
                else:
                    L[i][j] = 0
    return L.tolist()
    pass