import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as list of lists
    """
    # Your code here
    ans = []
    n = len(x)
    for i in range (n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h 
        x_minus[i] -= h 

        f_plus = np.array(f(x_plus))
        f_minus = np.array(f(x_minus))

        ans.append(((f_plus - f_minus)/(2*h)).tolist())
    ans = np.asarray(ans)
    return (np.transpose(ans)).tolist()
    pass