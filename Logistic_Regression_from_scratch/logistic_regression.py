import numpy as np

class LogisticRegression:
    def __init__(self,alpha = 0.001, no_of_iterations = 10000, tolerance_val = np.exp(-6)):
        self.alpha = alpha
        self.no_of_iterations = no_of_iterations
        self.tol = tolerance_val
        self.w = None
        self.b = None


    def fit(self,X,y):
        m,n = X.shape
        self.w = np.zeros(n)
        self.b = 0.
        prev_J = np.inf
        for i in range(self.no_of_iterations):
            z = self.predict_z(X)
            y_hat = self.sigmoid_fn(z)
            J_w_b = self.cost_funtion(y,y_hat)
            if np.abs(prev_J-J_w_b)<self.tol:
                return
            dj_dw,dj_db = self.gradient_calc(X,y,y_hat)
            self.w,self.b = self.gradient_descent(dj_dw,dj_db)
            prev_J=J_w_b
        return


    def predict_z(self,X):
        """
        input X shape must be m*n, m training examples and n number of features
        weight w shape must be (n,)
        b is a scalar
        """
        z = (X @ self.w) + self.b
        return z


    def sigmoid_fn(self,z):
        y_hat = 1/(1+(np.exp(-z)))
        return y_hat


    def cost_funtion(self,y,y_hat):
        m = y.shape[0]
        y_hat=np.clip(y_hat,a_min=0.001,a_max=0.999)
        J_w_b = (-np.sum((y*np.log(y_hat))+((1-y)*np.log(1-y_hat))))/m
        return J_w_b


    def gradient_calc(self,X,y,y_hat):
        m = X.shape[0]
        dj_dw = np.dot(X.T,(y_hat-y))/m
        dj_db = np.sum(y_hat-y)/m
        return dj_dw,dj_db


    def gradient_descent(self,dj_dw,dj_db):
        self.w -= self.alpha*dj_dw
        self.b -= self.alpha*dj_db
        return self.w,self.b



    def predict(self,X_test):
        if (self.w is None) or (self.b is None):
            raise ValueError("Please call fit function before calling predict method")
        z = self.predict_z(X_test)
        y_test = self.sigmoid_fn(z)
        prediction = (y_test>=0.5).astype(int)
        return prediction