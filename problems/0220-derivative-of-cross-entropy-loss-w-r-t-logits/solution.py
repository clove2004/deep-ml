import numpy as np
def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
    """
    Compute the derivative of cross-entropy loss with respect to logits.
    
    Args:
        logits: Raw model outputs (before softmax)
        target: Index of the true class (0-indexed)
        
    Returns:
        Gradient vector where gradient[i] = dL/d(logits[i])
    """
    # Your code here
    n = len(logits)
    q, ans = [], []
    summ = 0
    for i in logits:
        summ += np.exp(i)
    for i in logits:
        q.append((np.exp(i))/summ)
    q[target] -= 1
    return q

    pass