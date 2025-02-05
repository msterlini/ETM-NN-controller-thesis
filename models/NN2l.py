import torch.nn as nn

# Simple 2-layer (16-16) neural network with no output activation

# Activation function chosen is saturation
class NeuralNet(nn.Module):
  def __init__(self, input_size):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, 16)
    self.hardtanh1 = nn.Hardtanh()
    self.l2 = nn.Linear(16, 16)
    self.hardtanh2 = nn.Hardtanh()
    self.l3 = nn.Linear(16, 1)
  
    # Architecture
    self.arch = [16, 16]

  def forward(self, x):
    x = self.hardtanh1(self.l1(x))
    x = self.hardtanh2(self.l2(x))
    x = self.l3(x)
    return x