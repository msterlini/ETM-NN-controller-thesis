import os
import numpy as np

folder = 'deep_learning/2_layers/weights'

files = sorted(os.listdir(os.path.abspath(__file__ + "/../" + folder)))
W = []
b = []
for f in files:
  if f.startswith('W') and f.endswith('.csv'):
    W.append(np.loadtxt(os.path.abspath(__file__ + "/../" + folder + '/' + f), delimiter=','))
  elif f.startswith('b') and f.endswith('.csv'):
    b.append(np.loadtxt(os.path.abspath(__file__ + "/../" + folder + '/' + f), delimiter=','))

# Weights and biases reshaping
W[-1] = W[-1].reshape((1, len(W[-1])))

Wk = [np.load('deep_learning/K.npy')]
bk = [np.array([np.float32(0)])]

def nn(W, b, x):
  for id, weight in enumerate(W):
    x = np.clip(np.matmul(weight, x) + b[id], -1, 1)
  return x

state = np.random.uniform(-1, 1, 3)

print(nn(W, b, state*5))
print(nn(Wk, bk, state))