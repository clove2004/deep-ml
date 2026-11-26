import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    matrix = np.asarray(matrix)
    means = 0
    if mode == 'row':
        means = list(np.mean(matrix, axis = 1))
    else:
        means = list(np.mean(matrix, axis = 0))
    return means