import torch.nn as nn

# Simple 3-layer (32-32-32) neural network with no output activation

# Activation function chosen is saturation
class NeuralNet(nn.Module):
  def __init__(self, input_size):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, 32)
    self.hardtan1 = nn.Hardtanh()
    self.l2 = nn.Linear(32, 32)
    self.hardtan2 = nn.Hardtanh()
    self.l3 = nn.Linear(32, 32)
    self.hardtan3 = nn.Hardtanh()
    self.l4 = nn.Linear(32, 1)

    # Architecture
    self.arch = [32, 32, 32]
  
  def forward(self, x):
    x = self.hardtan1(self.l1(x))
    x = self.hardtan2(self.l2(x))
    x = self.hardtan3(self.l3(x))
    x = self.l4(x)
    return x