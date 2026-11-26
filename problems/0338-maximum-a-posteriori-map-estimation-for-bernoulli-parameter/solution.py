import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    one = 0
    for i in observations:
        if i == 1:
            one += 1
    n = len(observations)
    zero = n - one 
    x = alpha + one - 1
    y = beta + zero - 1
    return round(x/(x + y), 4)
    pass