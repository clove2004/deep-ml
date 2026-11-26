import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
    """
    Return the smallest rank k such that the top-k singular values of delta_W
    capture at least `energy_threshold` of the total squared-singular-value energy.
    """
    # Your code here
    U, S, Vt = np.linalg.svd(delta_W)
    E_total = sum(i**2 for i in S)
    if E_total == 0:
        return 0
    E_cur = 0
    for i in range (len(S)):
        E_cur += S[i]**2
        if E_cur >= E_total*energy_threshold:
            return i + 1

    pass