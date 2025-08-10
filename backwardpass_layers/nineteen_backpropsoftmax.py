
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
class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class CrossEntropyLoss(Loss):
    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape)==1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/samples

    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) ==1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_confidences = np.sum(y_pred_clipped* y_true,axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

class ActSoftMax_CategoricalCrossEntropy:
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = CrossEntropyLoss()

    def forward(self,inputs,y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output,y_true)

    def backward(self,dvalues,y_true):
        samples = len(dvalues)
        if len(y_true.shape) ==2:
            y_true = np.argmax(y_true,axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples),y_true] -=1
        self.dinputs = self.dinputs/samples

softmax_outputs = np.array([[0.7,0.1,0.2],[0.1,0.5,0.4],[0.02,0.9,0.08]])
class_targets = np.array([0,1,1])
softmax_loss = ActSoftMax_CategoricalCrossEntropy()
softmax_loss.backward(softmax_outputs,class_targets)
dvalues1 = softmax_loss.dinputs
print('Gradients : combined loss and activation')
print(dvalues1)