import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing
        self.py = None
        self.pdc0 = []
        self.pdc1 = []
        self.single_class = None

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        n_samples, n_features = X.shape
        N1 = np.sum(y)
        N0 = n_samples - N1
        if N1 == 0:
            self.single_class = 0
            return
        if N0 == 0:
            self.single_class = 1
            return
        self.py = N1/n_samples
        for j in range (n_features):
            c1 = [1 for i in range (n_samples) if y[i] == 1 and X[i][j] == 1]
            c0 = [1 for i in range (n_samples) if y[i] == 0 and X[i][j] == 1]
            c1 = np.asarray(c1)
            c0 = np.asarray(c0)
            self.pdc0.append((np.sum(c0) + self.smoothing)/(N0 + 2*self.smoothing))
            self.pdc1.append((np.sum(c1) + self.smoothing)/(N1 + 2*self.smoothing))
        self.pdc0 = np.asarray(self.pdc0, dtype=float)
        self.pdc1 = np.asarray(self.pdc1, dtype=float)

    def predict(self, X):
        # Predict class labels for test set X
        X = np.asarray(X, dtype=float)
        if self.single_class is not None:
            return np.full(X.shape[0], self.single_class, dtype=int)
        score_0 = np.log(1 - self.py) + X @ np.log(self.pdc0) + (1 - X) @ np.log(1 - self.pdc0)
        score_1 = np.log(self.py) + X @ np.log(self.pdc1) + (1 - X) @ np.log(1 - self.pdc1)
        return (score_1 > score_0).astype(int)