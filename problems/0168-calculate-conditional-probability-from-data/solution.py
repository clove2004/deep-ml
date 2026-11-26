def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    a, b = 0, 0
    for x1, y1 in data:
      if x1 == x:
        a += 1
        if y1 == y:
          b += 1
    if a == 0:
      return 0.0
    return round(b/a, 4)
    pass