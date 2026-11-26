import numpy as np
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    n = len(samples)
    samples = np.asarray(samples)
    val, counts = np.unique(samples, return_counts = True)
    ans = []
    for i in range (len(val)):
        ans.append((val[i], counts[i]/n))
    return ans
    pass