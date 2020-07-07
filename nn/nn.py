'''
Neural network with hidden layer
'''
from sklearn import datasets
import numpy as np
np.random.seed(0)
class myNeuralNetwork():
    def __init__(self):
        self.lr = 0.8
        # initialization
        self.weights = [np.random.randn(3,3),
                        np.random.randn(3,4),
                        np.random.randn(4,1)]
        pass
    def forward(self,x):
        self.z0 = x
        self.a1 = np.dot(self.z0,self.weights[0])
        self.z1 = self.sigmoid(self.a1)

        self.a2 = np.dot(self.z1,self.weights[1])
        self.z2 = self.sigmoid(self.a2)
        
        self.a3 = np.dot(self.z2,self.weights[2])
        self.z3 = self.sigmoid(self.a3)
        return self.z3

    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def error(self,x,y):
        yHat = model.forward(x)
        return (np.mean((yHat-y)**2))

    def backprop(self,x,y):
        self.forward(x)
        self.lr = 0.5
        
        
        self.d3 = (self.z3-y)*self.sigmoid_derivative(self.a3)
        
        da = self.d3*self.weights[2]
        db = np.matrix(self.sigmoid_derivative(self.a2)).T
        self.d2 = np.multiply(da,db)
        
        da = self.weights[1]*self.d2
        db = np.matrix(self.sigmoid_derivative(self.a1)).T
        self.d1 = np.multiply(da,db)

        
        self.dw3 = np.matrix(self.d3*self.z2).T
        self.dw2 = (self.d2*self.z1).T
        self.dw1 = (self.d1*self.z0).T

        self.weights[2] -= self.dw3
        self.weights[1] -= self.dw2
        self.weights[0] -= self.dw1
        
    def fit(self,x,y): 
        for _ in range(100):
            for idx in range(x.shape[0]):
                self.backprop(x[idx],y[idx])
            pass
        pass


# inputs
X = np.array([[1,1,1],[0,0,1]])
y = np.array([[1],[0]])

model = myNeuralNetwork()
print(model.error(X,y))
model.fit(X,y)
print(model.error(X,y))

X, y = (sklearn.datasets.make_moons(n_samples=10))

