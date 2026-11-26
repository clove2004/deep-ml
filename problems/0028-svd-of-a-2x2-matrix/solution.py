import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix.
    
    Args:
        A: 2x2 numpy array
    
    Returns:
        U: 2x2 orthogonal matrix (left singular vectors)
        s: 1D array of singular values
        V: 2x2 matrix (right singular vectors)
    """
    # Your code here
    y1, x1, y2, x2 = A[1][0] + A[0][1], A[0][0] - A[1][1], A[1][0] - A[0][1], A[0][0] + A[1][1]
    h1, h2 = np.sqrt(y1**2  + x1**2), np.sqrt(y2**2 + x2**2)
    o1, o2 = (h1 + h2)/2, abs(h1 - h2)/2
    alpha1, alpha2 = np.arctan2(y1, x1), np.arctan2(y2, x2)
    phi = (alpha2 + alpha1) / 2
    theta = (alpha2 - alpha1) / 2
    s = np.array([o1, o2])
    R = lambda x: np.array([[np.cos(x), -np.sin(x)],
                            [np.sin(x), np.cos(x)]])
    U = R(phi)
    Vt = R(theta)
    if h2 < h1:
        Vt[1, :] *= -1
    return (U, s, Vt)

    pass