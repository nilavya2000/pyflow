import numpy as np


class LogisticRegression:
    def __init__(self, alpha=0.01, iteration=1500, intercept=True):
        self.alpha=alpha
        self.iteration=iteration
        self.intercept=intercept
        
        

    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))

    def fit(self, x, y):
        self.m = len(x)
        self.n = np.size(x,1)
        if self.intercept :
            x = np.hstack((np.ones((len(x),1)), x))

        self.theta=np.zeros(x.shape[1])
                 
        for _ in range(self.iteration):
            self.z=np.dot(x, self.theta)
            self.h=self.sigmoid(self.z)
            grad=np.dot(x.T, (self.h-y)) / y.size
            self.theta = self.alpha * grad

         

    def loss(self, y):
        return ((-y*np.log(self.h))- ((1-y)*np.log(1-self.h))/self.m)

    def predict_prob(self, x):
        if self.intercept :
            x = np.hstack((np.ones((len(x),1)), x))
        return self.sigmoid(np.dot(x, self.theta))  

    def predict(self, x, threshold=0.5):
        return self.predict_prob(x)>= threshold




