import numpy as np

# Coding a single neuron

inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2.0
outputs = np.dot(weights,inputs) + bias
print(outputs)

### Coding a batch of neurons.

inp = [1,2,3,2.5]
weigh = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]

layer_out = np.dot(weigh,inp) +biases
print(layer_out)

## Coding with a batch of data.
inps = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
weis = [[0.2,0.8,-0.5,1],
        [0.5,-0.91,0.26,-0.5],
        [-0.26,-0.27,0.17,0.87]]
biass = [2.0,3.0,0.5]

outs = np.dot(inps,np.array(weis).T) + biass
print(outs)