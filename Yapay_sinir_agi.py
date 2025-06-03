import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)


input_vector = np.array([2.5, 3.0])

weights_input_hidden = np.array([
    [-1.4, 2.3],
    [0.7, -0.5]
])

bias_hidden = np.array([1.3, 0.9])

weights_hidden_output = np.array([[-0.1, -0.3]])

bias_output = np.array([0.5])

A = np.dot(weights_input_hidden,input_vector.T)

B = np.add(A,bias_hidden)

C = tanh(B)

D = np.dot(weights_hidden_output, C)

E = np.add(D,bias_output)

F = tanh(E)

print(E)
print(F)
