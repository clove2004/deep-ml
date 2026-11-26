import numpy as np

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    n = A.shape[0]
    M = np.hstack([A, b])
    tol = float(1e-10)
    for row in range (n):
        cur_r = row + np.argmax(np.abs(M[row:, row]))
        if np.abs(M[cur_r][row]) <= tol:
            return None 
        a = M[row].copy()
        M[row] = M[cur_r]
        M[cur_r] = a 
        for i in range (row + 1, n):
            fac = M[i][row]/M[row][row]
            M[i, row:] -= fac*M[row, row:]
    
    x = np.zeros(n, dtype = float)
    for i in range (n - 1, -1, -1):
        su = np.dot(M[i, i + 1:n], x[i + 1:n])
        x[i] = (M[i][n] - su)/M[i][i]

    return x
