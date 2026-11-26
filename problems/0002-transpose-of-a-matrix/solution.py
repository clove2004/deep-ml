import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    m, n = len(a), len(a[0])
    ans = np.zeros((n, m), dtype=int)
    for i in range (n):
        for j in range (m):
            ans[i][j] = a[j][i]
    return list(ans)
    pass