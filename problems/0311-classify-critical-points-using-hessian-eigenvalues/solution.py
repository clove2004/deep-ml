import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    tr, vec = np.linalg.eig(hessian)
    tr = tr.real
    if np.any(np.abs(tr) <= tol):
        return None
    ok1 = np.any(tr > tol)
    ok2 = np.any(tr < -tol)
    if ok1 and ok2:
        return 0
    elif ok2 == False:
        return -1
    else:
        return 1

    pass