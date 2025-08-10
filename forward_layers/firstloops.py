
inputs = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

outputs = []

for n_weights, biases in zip(weights,biases):
    neuron_out = 0
    for input, weight in zip(inputs,n_weights):
        neuron_out += input * weight
    neuron_out = neuron_out + biases
    outputs.append(neuron_out)

print(outputs)