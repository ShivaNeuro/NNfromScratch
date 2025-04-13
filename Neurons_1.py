inputs = [1,2,3,2.5]

weights = [[0.2,0.8,-0.5,1],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

weights1 = weights[0]
weights2 = weights[1]
weights3= weights[2]

biases = [2,3,0.5]
bias1 = biases[0]
bias2 = biases[1]
bias3 = biases[2]

outputs = [
    inputs[0]*weights1[0]+
    inputs[1]*weights1[1]+
    inputs[2]*weights1[2]+
    inputs[3]*weights1[3]+bias1,
    inputs[0]*weights2[0]+
    inputs[1]*weights2[1]+
    inputs[2]*weights2[2]+
    inputs[3]*weights2[3]+bias2,
    inputs[0]*weights3[0]+
    inputs[1]*weights3[1]+
    inputs[2]*weights3[2]+
    inputs[3]*weights3[3]+bias3
]

print(outputs)

################### Using Loops ###############
layer_output = []

for n_weights, n_biases in zip(weights,biases):
    out = 0
    for inp, weig in zip(inputs,n_weights):
        out += inp *weig
    out += n_biases
    layer_output.append(out)

print(layer_output)

#################################################3
