import numpy as np

class Lasso_Ridge_Regression:

    def __init__(self, lr=0.001, n_iters=1000, alpha=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.alpha = alpha  # regularization strength
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
             
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + self.alpha * np.sign(self.weights)
            '''
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + 2 * self.alpha * self.weights 
            
            this is for RidgeRegression
            
            '''
    
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
