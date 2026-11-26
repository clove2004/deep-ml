import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    matrix = np.asarray(matrix)
    symmetric = np.allclose(matrix, matrix.T)
    tr, vc = np.linalg.eig(matrix)
    ok = symmetric and np.all(tr.real > 1e-10) and np.allclose(tr.imag, 0)
    tr = tr.real
    tr = tr.tolist()
    for i in range (len(tr)):
        tr[i] = round(tr[i], 4)
    tr.sort()
    return {'is_positive_definite': ok, 'eigenvalues': tr}
    pass