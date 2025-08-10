
inputs = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

outputs =[inputs[0]*weights[0][0]+
          inputs[1]*weights[0][1]+
          inputs[2]*weights[0][2]+
          inputs[3]*weights[0][3]+biases[0],
          inputs[0]*weights[1][0]+
          inputs[1]*weights[1][1]+
          inputs[2]*weights[1][2]+
          inputs[3]*weights[1][3]+biases[1],
          inputs[0]*weights[2][0]+
          inputs[1]*weights[2][1]+
          inputs[2]*weights[2][2]+
          inputs[3]*weights[2][3]+biases[2]
          ]
print(outputs)