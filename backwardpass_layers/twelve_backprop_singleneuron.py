import numpy as np

weights = np.array([-3.0,-1.0,2.0])
bias = 1
inputs = np.array([1.0,-2.0,3.0])
target_output = 0.0
learning_rate = 0.001

def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return np.where(x>0,1.0,0.0)

for iteration in range(200):
    # Forward pass
    linear_output = np.dot(weights,inputs) + bias
    output = relu(linear_output)
    loss = (output - target_output)**2

    # Backward pass
    dloss_doutput = 2 * (output - target_output)
    doutput_dlinear = relu_derivative(linear_output)
    dlinear_dweights = inputs
    dlinear_dbias = 1

    dloss_dlinear = dloss_doutput * doutput_dlinear
    dloss_dweights = dloss_dlinear * dlinear_dweights
    dloss_dbias = dloss_dlinear * dlinear_dbias

    # Update weights and bias
    weights -= learning_rate * dloss_dweights
    bias -= learning_rate * dloss_dbias

    print(f"Iteration {iteration + 1}, Loss: {loss}")

print("Final weights:",weights)
print("Final bias:",bias)
