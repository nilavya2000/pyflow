import numpy as np
from collections import Counter
# euclidean distances
def e_dist(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X 
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict_(x) for x in X]
        return np.array(predicted_labels)

    def _predict_(self, x):
        distances = [e_dist(x, x_train) for x_train in self.X_train]    # compute distances 
        k_indices = np.argsort(distances)[:self.k] # nearest samples
        k_nearest_lables = [self.y_train[i] for i in k_indices] 
        most_common=Counter(k_nearest_lables).most_common(1) # common item 
        return most_common[0][0]