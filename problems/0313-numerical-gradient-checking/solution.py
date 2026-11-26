import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    numerical_grad = []
    n = len(x)
    for i in range (n):
        xc_plus = x.copy()
        xc_minus = x.copy()

        xc_plus[i] += epsilon
        xc_minus[i] -= epsilon
        numerical_grad.append((f(xc_plus) - f(xc_minus))/(2*epsilon))
    numerical_grad = np.asarray(numerical_grad)
    relative_error = np.linalg.norm((numerical_grad - analytical_grad),ord=2)/(np.linalg.norm(numerical_grad,ord=2) + np.linalg.norm(analytical_grad,ord=2))
    return (numerical_grad, relative_error)
    pass