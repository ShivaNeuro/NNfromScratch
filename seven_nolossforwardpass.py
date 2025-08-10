import numpy as np
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()

class Dense_Layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.bias = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) +self.bias

class Activation_Relu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        prob = exp/np.sum(exp,axis=1,keepdims=True)
        self.output = prob

X,Y = spiral_data(samples=100,classes=3)
dense_layer1 = Dense_Layer(2,3)
relu = Activation_Relu()
dense_layer2 = Dense_Layer(3,3)
soft = Activation_Softmax()
dense_layer1.forward(X)
relu.forward(dense_layer1.output)
dense_layer2.forward(relu.output)
soft.forward(dense_layer2.output)
print(soft.output)