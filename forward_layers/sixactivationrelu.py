############RELU##########################
import numpy as np
#inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
#output = np.maximum(0,inputs)
#print(output)

######## SOFT MAX ##########################3
#inputs = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
#exp_vals = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
#prob = exp_vals/np.sum(exp_vals,axis=1,keepdims=True)
#print(prob)
#print(np.sum(prob,axis=1))
#######################################################3

from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) +self.biases
class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backwards(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <=0]=0

class Activation_Softmax:
    def forward(self,inputs):
        exp_vals = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        prob = exp_vals/np.sum(exp_vals,axis=1,keepdims=True)
        self.output = prob

X,Y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,3)
dense1.forward(X)
act = Activation_ReLU()
act.forward(dense1.output)
act_sfmax = Activation_Softmax()
act_sfmax.forward(act.output)
print(act_sfmax.output)



