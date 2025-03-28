from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CamembertWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, P_train=None, P_val=None):
        self.P_train = P_train
        self.P_val = P_val

    def fit(self, X, y): return self

    def predict_proba(self, X):
        return self.P_train if len(X) == len(self.P_train) else self.P_val

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class ResNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, P_train=None, P_val=None):
        self.P_train = P_train
        self.P_val = P_val

    def fit(self, X, y): return self

    def predict_proba(self, X):
        return self.P_train if len(X) == len(self.P_train) else self.P_val

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
