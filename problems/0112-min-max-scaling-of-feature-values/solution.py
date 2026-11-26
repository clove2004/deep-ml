def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    mx, mn = max(x), min(x)
    if mx == mn:
        return [0.0 for _ in range (len(x))]
    for i in range (len(x)):
        x[i] = (x[i] - mn)/(mx - mn)
    return x