from stable_baselines3 import PPO
import pandas as pd
import torch.nn as nn

def extract_parameters(model_path: str):
  model = PPO.load(model_path)

  weight_and_biases = {}

  for name, param in model.policy.named_parameters():
    if 'policy' in name or 'action' in name:
      if 'weight' in name:
        weight_and_biases[name] = param.detach().numpy()
      elif 'bias' in name:
        weight_and_biases[name] = param.detach().numpy()
        
  for name, value in weight_and_biases.items():
    df = pd.DataFrame(value)
    filename = f'./new_weights/{name}.csv'
    df.to_csv(filename, index=False, header=False)  

def get_structure()