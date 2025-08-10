import numpy as np

inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2

outputs = np.dot(inputs,weights) +bias
print(outputs)

############################################

inp = [1,2,3,2.5]
weis = [[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
bia = [2.0,3.0,0.5]

lay_out = np.dot(weis,inp)+bia
print(lay_out)

###################################3333333333
inps = [[1,2,3,2.5],[2,5,-1,2],[-1.5,2.7,3.3,-0.8]]
weigs = [[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]
bi = [2.0,3.0,0.5]

lay_out = np.dot(inps,np.array(weigs).T)+bi
print(lay_out)
