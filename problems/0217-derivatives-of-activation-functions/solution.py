import numpy as np
def sigmoid (x):
    return 1.0/(1 + np.exp(-x))
def dsigmoid (x):
    return sigmoid(x)*(1 - sigmoid(x))
def tanh(x):
    return np.tanh(x)
def dtanh(x):
    return 1 - tanh(x)**2
def relu(x):
    return np.maximum(x, 0)
def drelu(x):
    if x > 0:
        return 1
    else:
        return 0
def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here
    return {'sigmoid': dsigmoid(x), 'tanh': dtanh(x), 'relu': drelu(x)}
    pass