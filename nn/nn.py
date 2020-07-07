'''
Neural network with hidden layer
'''

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
        
        self.a1 = np.dot(x,self.weights[0])
        self.z1 = self.sigmoid(self.a1)
        
        self.a2 = np.dot(self.z1,self.weights[1])
        self.z2 = self.sigmoid(self.a2)
        
        self.a3 = np.dot(self.z2,self.weights[2])
        self.z3 = self.sigmoid(self.a3)
        return self.z3
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return sigmoid(x)/(1-sigmoid(x))
    
    def fit(self,x,y):
        self.forward(x)
        self.d3 = -(y-self.z3)*sigmoid_derivative(self.a3)
        self.d2 = self.d3*self.weights[1]*self.sigmoid_derivative(self.a2)
        
        print(np.matrix(self.d3))
        print(np.matrix(self.z2).T)
        print("="*4)
        print(np.multiply( np.matrix(self.d3), np.matrix(self.z2).T ))
        
        print(np.multiply( np.matrix(self.d2), np.matrix(self.z1).T ))
        
        '''
        self.dWd3 = np.multiply(np.matrix(self.d3),np.matrix(self.z2).T)
        self.dWd2 = np.multiply(self.z1,np.matrix(self.d2).T)
        self.weights[2] += self.dWd3
        self.weights[1] += self.dWd2
        '''
        
# inputs
X = np.array([[1,1,1],[0,0,1]])
y = np.array([[1],[0]])

model = myNeuralNetwork()
model.fit(X[0],y[0])
