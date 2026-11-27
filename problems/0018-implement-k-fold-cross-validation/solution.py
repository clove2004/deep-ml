import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    indices = [i for i in range(n_samples)]
    if shuffle == True:
        np.random.shuffle(indices)
    sz = [int(n_samples//k) for _ in range (k)]
    r = n_samples%k 
    for i in range (r):
        sz[i] += 1
    a, t = [], 0
    ans = []
    for i in range (k):
        a.append(indices[t:t + sz[i]])
        t += sz[i]
    for i in range (k):
        test = a[i]
        train = []
        for j in range (k):
            if i != j:
                train = train + a[j]
        ans.append((train, test))
    return ans