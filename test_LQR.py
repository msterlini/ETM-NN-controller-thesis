from LMI import LMI
import os
import numpy as np
from system import System

W = [np.load('deep_learning/K.npy')]
b = [np.array([np.float32(0)])]

# sys = System(W, b, [], 0, 0)

folder = 'deep_learning/3_layers/weights'
  # folder = 'weights'

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

lmi = LMI(W, b)

alpha = lmi.search_alpha(1.0, 0.0, 1e-3, verbose=True)
