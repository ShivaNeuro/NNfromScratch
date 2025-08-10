import numpy as np

A =[[1,2,3],[4,5,6],[7,8,9]]
print(np.sum(A))

print(np.sum(A, axis=None))
print(np.sum(A,axis=None).shape)

print(np.sum(A, axis=0))
print(np.sum(A,axis=0).shape)

print(np.sum(A, axis=1))
print(np.sum(A,axis=1).shape)

print(np.sum(A, axis=0,keepdims=True))
print(np.sum(A,axis=0,keepdims=True).shape)

print(np.sum(A, axis=1,keepdims=True))
print(np.sum(A,axis=1,keepdims=True).shape)

