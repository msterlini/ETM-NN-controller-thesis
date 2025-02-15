from models.NN6l import NeuralNet
import torch
import pandas as pd
import os

input_size = 3
model = NeuralNet(input_size)
model.load_state_dict(torch.load('model.pth'))

weight_and_biases = {}

for name, param in model.named_parameters():
  if 'weight' in name:
    weight_and_biases[name] = param.detach().numpy()
  elif 'bias' in name:
    weight_and_biases[name] = param.detach().numpy()

weight_count = 1
bias_count = 1

if not os.path.exists('weights'):
  os.makedirs('weights')
  
for name, value in weight_and_biases.items():
  df = pd.DataFrame(value)
  if 'weight' in name:
    filename = f'./weights/W{weight_count}.csv'
    weight_count += 1
  elif 'bias' in name:
    filename = f'./weights/b{bias_count}.csv'
    bias_count += 1
  df.to_csv(filename, index=False, header=False)