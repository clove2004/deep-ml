import numpy as np
def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.
    
    Args:
        x: Input vector of real numbers
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here
    n = len(x)
    su = 0
    sof = []
    for i in x:
        su += np.exp(i)
    for i in x:
        sof.append((np.exp(i))/su)
    J = np.zeros((n, n), dtype = float)
    for i in range (n):
        for j in range (n):
            if i != j:
                J[i][j] = -sof[i]*sof[j]
            else:
                J[i][j] = sof[i]*(1 - sof[i])
    return J.tolist()
    pass