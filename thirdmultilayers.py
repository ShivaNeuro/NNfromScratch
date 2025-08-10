import numpy as np

inputs=[[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
weights = [[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
biases = [2,3,0.5]

weights2 = [[0.1,-0.14,0.5],[-0.5,0.12,-0.33],[-0.44,0.73,-0.13]]
biases2 = [-1,2,-0.5]

inp = np.array(inputs)
weight1 = np.array(weights)
bias1 = np.array(biases)
weight2 = np.array(weights2)
bias2 = np.array(biases2)

lay1_output = np.dot(inp,weight1.T)+bias1
lay2_output = np.dot(lay1_output,weight2.T)+bias2
print(lay1_output)
print(lay2_output)