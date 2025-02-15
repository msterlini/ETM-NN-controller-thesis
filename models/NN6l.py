import torch.nn as nn

# Simple 6-layer (32-32-32-32-32-32) neural network with no output activation

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
    self.l4 = nn.Linear(32, 32)
    self.hardtan4 = nn.Hardtanh()
    self.l5 = nn.Linear(32, 32)
    self.hardtan5 = nn.Hardtanh()
    self.l6 = nn.Linear(32, 32)
    self.hardtan6 = nn.Hardtanh()
    self.l7 = nn.Linear(32, 1)

    # Architecture
    self.arch = [32, 32, 32, 32, 32, 32]
  
  def forward(self, x):
    x = self.hardtan1(self.l1(x))
    x = self.hardtan2(self.l2(x))
    x = self.hardtan3(self.l3(x))
    x = self.hardtan4(self.l4(x))
    x = self.hardtan5(self.l5(x))
    x = self.hardtan6(self.l6(x))
    x = self.l7(x)
    return x