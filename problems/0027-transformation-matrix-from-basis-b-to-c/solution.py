import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B, C = np.asarray(B), np.asarray(C)
    P = np.linalg.inv(C) @ B 
    P = np.round(P, 4)
    return P