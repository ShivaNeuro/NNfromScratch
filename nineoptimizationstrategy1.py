import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data
nnfs.init()

x,y = vertical_data(samples=100,classes=3)
plt.scatter(x[:,0],x[:,1],c=y,s=40,cmap='brg')
plt.show()

class DenseLayer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.bias

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
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        if len(y_true.shape) ==1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) ==2:
            print("One Hot")
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

dense1 = DenseLayer(2,3)
act1 = Activation_Relu()
dense2 = DenseLayer(3,3)
act2 = Activation_Softmax()
loss_function = CrossEntropyLoss()

lowest_loss = 99999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.bias.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.bias.copy()

for iteration in range(100000):
    dense1.weights = 0.05 * np.random.randn(2,3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.bias = 0.05 * np.random.randn(1, 3)
    dense1.forward(x)
    act1.forward(dense1.output)
    dense2.forward(act1.output)
    act2.forward(dense2.output)
    loss = loss_function.calculate(act2.output,y)
    prediction = np.argmax(act2.output,axis=1)
    accu = np.mean(prediction == y)
    if loss < lowest_loss:
        print("new set of weights found , iteration :",iteration,'loss:',loss,'acc',accu)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.bias.copy()
        lowest_loss = loss






