import numpy as np
import random as rd
import matplotlib.pyplot as plt

# Generates random weight
wt = lambda: rd.random()

# Function which helps us minimize our error
def sigmoid(x, drv=False):
    f = 1.0/(1.0 + np.exp(-x))
    if drv:
        return f*(1 - f)
    return f

# Neural Network Class
class NeuralNet:

    # Initialize sizes and epochs
    def __init__(self, in_size=0, out_size=0, epochs=150):
        self.in_size = in_size
        self.out_size = out_size
        self.epochs = epochs

    # Train neural net
    def __call__(self, inputs, outputs):
        close = False
        for epoch in range(self.epochs):
            # Forward Propigation
            for i in self.axis:
                if i == self.axis[0]:
                    self.layers[i] = self.weights[i].T.dot(inputs)
                    self.slayers[i] = sigmoid(self.layers[i])
                else:
                    self.layers[i] = self.weights[i].T.dot(self.slayers[i+1])
                    self.slayers[i] = sigmoid(self.layers[i])

            # Backwards Propigation
            for i in self.axis2:
                if i == self.axis2[0]:
                    error = pow(outputs - self.slayers[i], 2)
                    print(error)
                    if np.sum(error) < 0.001:
                        close = True
                    delta = 2*(outputs - self.slayers[i])*sigmoid(self.layers[i], drv=True)
                    self.weights[i] += delta
                else:
                    error = self.weights[i-1].dot(delta)
                    delta = error*sigmoid(self.layers[i],drv=True)
                    self.weights[i] += delta

            # Ends training when error level has been met
            if close == True:
                break
                
    # Builds weights and layers
    def build(self):
        i, j = self.in_size, self.out_size
        self.axis = list(range(i, j, -1))
        self.axis2 = self.axis[::-1]

        self.weights = {}
        self.layers = {}
        self.slayers = {}

        for k in self.axis:
            self.weights[k] = np.random.rand(k, k-1)
            self.layers[k] = np.zeros(k-1)
            self.slayers[k] = np.zeros(k-1)

    # Tests the result of the Neural Net
    def testNN(self, inputs, outputs):
        for i in self.axis:
            if i == self.axis[0]:
                self.layers[i] = self.weights[i].T.dot(inputs)
                self.slayers[i] = sigmoid(self.layers[i])
            else:
                self.layers[i] = self.weights[i].T.dot(self.slayers[i+1])
                self.slayers[i] = sigmoid(self.layers[i])

        dn = self.dnorm(self.slayers[self.axis[-1]], outputs)
        error = np.sqrt(np.sum([(i - j)**2 for i, j in zip(dn, outputs)]))
        print('Error: ', error)
    
    # Normalize the data
    def norm(self, x):
        m0, m1 = min(x), max(x)
        return (x - m0)/(m1 - m0)

    # Un-normailze the data
    def dnorm(self, x, y):
        m0, m1 = min(x), max(x)
        return y*(m1 - m0) + m0
            
# Call NeuralNet class object
nnet = NeuralNet(in_size=10, out_size=3)

# Call build weights
nnet.build()

# Sample inputs and outputs
inputs = np.array([5, 6, 3, 1, 2, 5, 9, 8, 1, 4])
outputs = np.array([2, 9, 4])

# Normalized
nIN = nnet.norm(inputs)
nOUT = nnet.norm(outputs)

# Train
nnet(nIN, nOUT)

# Test
nnet.testNN(nIN, outputs)
