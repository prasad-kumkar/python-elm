import numpy as np

class ELM:
    def __init__(self, num_layers=100):
        self.num_layers = num_layers
        self.W = None
        self.inverse = None

    def input_to_hidden(self, x):
        a = np.dot(x, self.W)
        a = np.maximum(a, 0, a) # ReLU
        return a

    def fit(self, xtrain, ytrain):
        self.W = np.random.normal(size=[xtrain.shape[1], self.num_layers])
        H = self.input_to_hidden(xtrain)
        Ht = np.transpose(H)
        self.inverse = np.dot(np.linalg.inv(np.dot(Ht, H)), np.dot(Ht, ytrain))

    def predict(self, xtest):
        return np.dot(self.input_to_hidden(xtest), self.inverse) 
