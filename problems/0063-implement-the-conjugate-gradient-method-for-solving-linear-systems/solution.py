import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    """
    Solve the system Ax = b using the Conjugate Gradient method.

    :param A: Symmetric positive-definite matrix
    :param b: Right-hand side vector
    :param n: Maximum number of iterations
    :param x0: Initial guess for solution (default is zero vector)
    :param tol: Convergence tolerance
    :return: Solution vector x
    """
    # Your code here
    A, b = np.asarray(A), np.asarray(b)
    lap = n 
    n = len(A)
    if x0 is None: 
        x0 = np.zeros(n, dtype=float)
    r0 = b - (A @ x0)
    p0 = r0
    for _ in range (lap):
        a0 = (r0 @ r0)/(p0 @ A @ p0)
        x1 = x0 + a0*p0 
        r1 = r0 - a0*(A @ p0)
        if np.abs(np.linalg.norm(r1)) < tol:
            r0, p0, x0 = r1, p1, x1
            break
        b0 = (r1 @ r1)/(r0 @ r0)
        p1 = r1 + b0*p0
        r0, p0, x0 = r1, p1, x1
    return x0
    pass
