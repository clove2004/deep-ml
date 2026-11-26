import numpy as np
def sigmoid (x):
    return (1.0/(1 + np.exp(-x)))
def dsigmoid (x):
    return sigmoid(x)*(1 - sigmoid(x))
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    # Your code here
    mse_values = []
    n_samples, n_features  = features.shape
    for _ in range (epochs):
        z = (features @ initial_weights) + initial_bias
        ms = 0
        for j in range (n_features):
            curw = 0
            for i in range (n_samples):
                curw += 2*(sigmoid(z[i]) - labels[i])*dsigmoid(z[i])*features[i][j]
            initial_weights[j] -= learning_rate*curw/n_samples 
        curb = 0
        for i in range (n_samples):
            ms += (sigmoid(z[i]) - labels[i])**2
            curb += 2*(sigmoid(z[i]) - labels[i])*dsigmoid(z[i])
        initial_bias -= learning_rate*curb/n_samples

        mse_values.append(ms/n_samples)

    return initial_weights, initial_bias, mse_values