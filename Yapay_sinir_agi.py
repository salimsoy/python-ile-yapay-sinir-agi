import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Kullanıcıdan parametre alalım
input_vector = np.array([2.5, 3.0])  # Girdi verisi (örnek)

# Gizli katman ağırlıkları (2 giriş -> 3 nöron)
weights_input_hidden = np.array([
    [-1.4, 2.3],
    [0.7, -0.5]
])

# Gizli katman bias
bias_hidden = np.array([1.3, 0.9])

# Çıkış katmanı ağırlıkları (3 gizli -> 1 çıkış)
weights_hidden_output = np.array([[-0.1, -0.3]])

# Çıkış bias
bias_output = np.array([0.5])

A = np.dot(weights_input_hidden,input_vector.T)

B = np.add(A,bias_hidden)

C = tanh(B)

D = np.dot(weights_hidden_output, C)

E = np.add(D,bias_output)

F = tanh(E)

print(E)
print(F)
