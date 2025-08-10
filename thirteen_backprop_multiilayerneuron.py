import numpy as np

weights = np.array([[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1,1.1,1.2]])
bias = np.array([0.1,0.2,0.3])
inputs = np.array([1,2,3,4])
target_output = 0.0
learning_rate = 0.001

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x>0,1.0,0.0)

for iteration in range(200):
    # Forward pass
    z = np.dot(weights,inputs) + bias
    a = relu(z)
    y = np.sum(z)

    #loss
    loss = y ** 2

    # Backward pass
    #Gradient of loss with respect to y
    dl_dy = 2 * y

    # Gradient of y with respect to a
    dy_da = np.ones_like(a)

    # Gradient of a with respect to z (Relu derivativE)
    da_dz = relu_derivative(z)

    # Gradeint of loss with respect to z
    dl_dz = dl_dy * da_dz

    # Gradient of Loss with respect to weights and biases
    dL_dW = np.outer(dl_dz,inputs)
    dl_db = dl_dz

    weights -= learning_rate * dL_dW
    bias -= learning_rate * dl_db
    print("Iteration",iteration+1)
    print("Updated loss ",loss)
print("Final Updated weights",weights)
print("Final Updated bias",bias)
