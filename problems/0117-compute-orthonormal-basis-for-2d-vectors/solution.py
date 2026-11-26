import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    ans = []
    vectors = np.asarray(vectors, dtype=float)
    n, m = vectors.shape
    for i in range (n):
        ui = vectors[i]/np.linalg.norm(vectors[i])
        wk = vectors[i].copy()
        for vec in ans:
            wk -= (np.dot(vec, vectors[i]))*vec
        if np.linalg.norm(wk) > tol:
            ui = wk/np.linalg.norm(wk)
            ans.append(ui)
    return ans
    pass