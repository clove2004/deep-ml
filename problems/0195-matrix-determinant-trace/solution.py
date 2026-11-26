import numpy as np
def matrix_determinant_and_trace(matrix: list[list[float]]) -> tuple[float, float]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) represented as list of lists
    
    Returns:
        Tuple of (determinant, trace)
    """
    # Your code here
    matrix = np.asarray(matrix)
    dete = np.linalg.det(matrix)
    trc = 0
    for i in range (len(matrix)):
        trc += matrix[i][i]
    return (dete, trc)
    pass