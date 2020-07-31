import numpy as np

class LinearRegression:
    def __init__(self, x, y , alpha=0.3, iteration=1500):
        self.alpha = alpha 
        self.iteration = iteration
        self.m = len(y)
        self.features = np.size(x, 1)
        self.x = np.hstack((np.ones((self.m,1)), (x-np.mean(x,0))/np.std(x, 0)))
        self.y = y[:, np.newaxis]
        self.theta = np.zeros((self.features+1, 1))
        self.coeff=None
        self.interccept=None


    def fit(self):
        for _ in range(self.iteration):
            self.theta = self.theta - (self.alpha/self.m)*self.x.T @ (self.x @ self.theta - self.y)
        self.interccept=self.theta[0]
        self.coeff=self.theta[1:]
        return self
    
    def theta_value(self):
        return self.theta
    
    
    def predict(self, x):
        samples=np.size(x,0)
        y = np.hstack((np.ones((samples, 1)), (x-np.mean(x, 0)) / np.std(x, 0))) @ self.theta
        return y


    def score(self, x=None, y=None):
        if x is None :
            x=self.x
        else:
            samples=np.size(x,0) 
        
        if y is None:
            y=self.y
        else:
            y=y[:,np.newaxis]
        
        y_pred = x @ self.theta
        score = 1 - (((y-y_pred)**2).sum() / ((y-np.mean(y))**2).sum())
        return score
