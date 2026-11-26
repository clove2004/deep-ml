import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Your code here
    M = np.array(A, dtype=float, copy=True)
    if M.size == 0:
        return 0
    m, n = M.shape
    rank = 0
    for col in range (n):
        if rank >= m:
            break
        cur_row = rank + np.argmax(np.abs(M[rank:, col]))
        if(np.abs(M[cur_row][col]) <= tol):
            continue
        if cur_row != rank:
            a = M[rank].copy()
            M[rank] = M[cur_row]
            M[cur_row] = a 
        for row in range (rank + 1, m):
            fac = M[row][col]/M[rank][col]
            M[row, col:] -= fac*M[rank, col:]
        rank += 1
    return rank
    pass