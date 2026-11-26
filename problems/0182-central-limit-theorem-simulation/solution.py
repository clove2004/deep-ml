import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    # Your implementation here
    X, true_mean, true_std = 0, 0, 0
    if distribution == 'uniform':
        X = np.random.uniform(0, 1, size=(runs, n))
        true_mean, true_std = 0.5, np.sqrt(1/12)
    elif distribution == 'exponential':
        X = np.random.exponential(1.0, size=(runs, n))
        true_mean, true_std = 1.0, 1.0
    elif distribution == 'bernoulli':
        X = (np.random.rand(runs, n) < 0.3).astype(float)
        true_mean, true_std = 0.3, np.sqrt(0.3*0.7)
    else:
        raise ValueError
    z_score = []
    for i in X:
        z_score.append((np.mean(i) - true_mean)/(true_std/np.sqrt(n)))
    z_score = np.asarray(z_score)
    return {'mean': np.mean(z_score), 'std': np.std(z_score)}
    pass