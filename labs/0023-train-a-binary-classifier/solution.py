# --- cell 1 ---
import numpy as np
# You can import any sklearn module you need
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # TODO: train a model and return a predict function
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=int).ravel()
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=int).ravel()

    models = {
        "SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=42)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="uniform")),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    }
    best_model = models["SVM"]
    best_score = -1.0
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = accuracy_score(y_val, model.predict(X_val))
        if score > best_score:
            best_score = score
            best_model = model

    final_model = clone(best_model)
    final_model.fit(np.concatenate((X_train, X_val), axis=0), np.concatenate((y_train, y_val), axis=0))

    def predict(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return final_model.predict(X).astype(int).ravel()
    return predict