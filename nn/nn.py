'''
Neural network with hidden layer
'''
from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib
import math
matplotlib.use('Agg')

import numpy as np
np.random.seed(0)



class myNeuralNetwork():
    def __init__(self):
        self.lr = 0.8
        # initialization
        self.weights = [np.random.randn(3,4),
                        np.random.randn(4,5),
                        np.random.randn(5,1)]
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
        self.lr = 0.1
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

        self.weights[2] -= self.dw3*self.lr
        self.weights[1] -= self.dw2*self.lr
        self.weights[0] -= self.dw1*self.lr
        
    def fit(self,x,y): 
        for epoch in range(800):
            for idx in range(x.shape[0]):
                self.backprop(x[idx],y[idx])
            if epoch%10 == 0:
                self.plotMesh(x,y,epoch)
            pass
        pass

    def plotMesh(self,x,y,epoch=0):
        x_ = np.arange(-2, 3, 0.1)
        y_ = np.arange(-2, 3, 0.1)
        xx, yy = np.meshgrid(x_, y_)
        xx,yy = xx.reshape(-1,1), yy.reshape(-1,1)
        # Forward meshgrid

        Xb = np.hstack([xx,yy,np.ones((yy.shape[0],1))])
        square = int(math.sqrt(Xb.shape[0])) # for converting dimension to 2d
        
        z1 = model.forward(Xb).reshape(square,square)
        plt.contourf(x_,y_,z1,cmap='RdBu')
        plt.scatter(x[:,0],x[:,1],s=5,c=y,cmap='RdBu')
        plt.xlim(-2,3)
        plt.ylim(-2,3)
        plt.savefig("images/epoch-{}.png".format(epoch))
        


# inputs

#Xb = np.array([[1,1,1],[0,0,1]])
#y = np.array([[1],[0]])

X, y = (datasets.make_moons(n_samples=100,noise=0.2, random_state=0))
b = np.ones((X.shape[0],1))
Xb = np.hstack([X,b])

model = myNeuralNetwork()


e1 = model.error(Xb,y)
model.fit(Xb,y)
e2 = model.error(Xb,y)
print("{} -> {}".format(e1,e2))

model.plotMesh(X,y)
plt.show()


'''
for plotting:
'''
