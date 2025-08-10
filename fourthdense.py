from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
#X,y = spiral_data(samples=100,classes=3)
#plt.scatter(X[:,0],X[:,1])
#plt.show()
#plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
#plt.show()

#===========================================

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
        print(self.biases)

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) +self.biases

X,Y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,3)
dense1.forward(X)
print(dense1.output[:5])
