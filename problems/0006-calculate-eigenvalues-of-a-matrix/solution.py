import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    matrix = np.asarray(matrix)
    tr, vec = np.linalg.eig(matrix)
    return tr.tolist()