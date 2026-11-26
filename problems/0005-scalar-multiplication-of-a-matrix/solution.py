import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    # Your code here
    ma = np.asarray(matrix)
    return (ma*scalar).tolist()
    pass