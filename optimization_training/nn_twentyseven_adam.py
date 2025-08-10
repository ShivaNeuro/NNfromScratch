from pickletools import optimize

from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases

    def backward(self,dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        self.dinputs = np.dot(dvalues,self.weights.T)

class Activation_Relu:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0]=0

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

class Optimizer_GD:
    def __init__(self,learning_rate = 0.5):
        self.learning_rate = learning_rate

    def update_params(self,layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

class Optimizer_GDdecay:
    def __init__(self,learning_rate = 0.5,decay=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1.+self.decay * self.iterations))

    def update_params(self,layer):
        layer.weights += -self.current_learning_rate * layer.dweights
        layer.biases += -self.current_learning_rate * layer.dbiases

    def post_update_params(self):
        self.iterations +=1

class Optimizer_GDdecayMomentum:
    def __init__(self,learning_rate = 0.5,decay=0,momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1.+self.decay * self.iterations))

    def update_params(self,layer):
        if self.momentum:
            if not hasattr(layer,'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate *layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        layer.weights +=weight_updates
        layer.biases +=bias_updates

    def post_update_params(self):
        self.iterations +=1

class Optimizer_RMSProp:
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache = self.rho  * layer.weight_cache + (1-self.rho) *layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho) *layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

class Optimizer_Adagrad:
    def __init__(self, learning_rate = 1., decay = 0, epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate *(1./(1.+self.decay * self.iterations))

    def update_params(self,layer):
        if not hasattr(layer,'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations +=1


class Optimizer_Adam:
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta_1 = 0.9,beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1./(1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1-self.beta_1)*layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1-self.beta_1)* layer.dbiases
        layer.weight_cache = self.beta_2  * layer.weight_cache + (1-self.beta_2) *layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2) *layer.dbiases ** 2
        weight_momentums_corrected = layer.weight_momentums /(1-self.beta_1**(self.iterations+1))
        bias_momentums_corrected = layer.bias_momentums/(1-self.beta_1**(self.iterations+1))
        weight_cache_corrected = layer.weight_cache/(1-self.beta_2**(self.iterations+1))
        bias_cache_corrected = layer.bias_cache/(1-self.beta_2**(self.iterations+1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1




X,y = spiral_data(samples=100,classes=3)
dense1 = Layer_Dense(2,64)
activation1 = Activation_Relu()
dense2 = Layer_Dense(64,3)
loss_act = ActSoftMax_CategoricalCrossEntropy()
#optimizer = Optimizer_GD()
#optimizer = Optimizer_GDdecay(decay = 1e-3)
#optimizer = Optimizer_GDdecayMomentum(decay=1e-3,momentum=0.5)
#optimizer = Optimizer_Adagrad(decay=1e-4)
#optimizer = Optimizer_RMSProp(learning_rate=0.02,decay=1e-5,rho=0.999)
optimizer = Optimizer_Adam(learning_rate=0.02,decay=1e-5)
for epoch in range(10001):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_act.forward(dense2.output,y)
    predictions = np.argmax(loss_act.output,axis=1)
    if len(y.shape) ==2:
        y = np.argmax(y,axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 50:
        print(f'epoch :{epoch} ' + f'acc:{accuracy:.3f},' + f'loss: {loss:.3f}' + f' lr : {optimizer.current_learning_rate}')
    # Backward pass
    loss_act.backward(loss_act.output,y)
    dense2.backward(loss_act.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update of weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

