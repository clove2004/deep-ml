import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
    """
    Compute entropy of P and cross-entropy between P and Q.
    
    Args:
        P: True probability distribution
        Q: Predicted probability distribution
    
    Returns:
        Tuple of (entropy H(P), cross-entropy H(P,Q))
    """
    # Your code here
    hp, hpq = 0, 0
    for i in range (len(P)):
        if P[i] != 0:
            hp -= P[i]*np.log(P[i])
        if Q[i] != 0:
            hpq -= P[i]*np.log(Q[i])
    return (round(hp, 6), round(hpq, 6))
    pass