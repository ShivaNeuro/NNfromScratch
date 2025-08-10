import numpy as np

################ CLASS TARGETS NUMBERS###############333333
softmax_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])
class_targets = [0,1,1]
print(softmax_outputs[[0,1,2],class_targets])

log_val = -np.log(softmax_outputs[[range(len(softmax_outputs))],class_targets])
avg_loss = np.mean(log_val)
print(avg_loss)

############# CLASS TARGETS ONE HOT ENCODED ################
soft_outputs = np.array([[0.7,0.1,0.2],
                            [0.1,0.5,0.4],
                            [0.02,0.9,0.08]])
class_targets = np.array([[1,0,0],[0,1,0],[0,1,0]])
out = np.sum(soft_outputs * class_targets,axis=1)
print(out)
log_val = -np.log(out)
avg_loss = np.mean(log_val)
print(avg_loss)