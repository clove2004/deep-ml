import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n = data.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_num = int(train_frac*n)
    valid_num = int(validation_frac*n)
    train_data = data[indices[:train_num]]
    valid_data = data[indices[train_num:(train_num + valid_num)]]
    test_data = data[indices[(train_num + valid_num):]]
    return [train_data, valid_data, test_data]