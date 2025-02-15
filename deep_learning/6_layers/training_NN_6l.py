import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from models.NN6l import NeuralNet

# Load dataset
df = pd.read_csv('../dataset.csv')
states = df[['theta', 'vtheta', 'int']].values
inputs = df[['u']].values

# Convert to PyTorch tensors
states_tensor = torch.tensor(states, dtype=torch.float32)
input_tensor = torch.tensor(inputs, dtype=torch.float32)

# Create dataset and dataloader
dataset = TensorDataset(states_tensor, input_tensor)

# Batch size
batch_size = 1024
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model, loss and optimizer 
input_size = 3
model = NeuralNet(input_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ========== Training ==========
# Number of epochs
num_epochs = 10
# Loop over the dataset multiple times
loss_values = []
for epoch in range(num_epochs):
  
  for states, inputs in dataloader:
    # Move tensors to the configured device
    states, inputs = states.to(device), inputs.to(device)
    # Forward pass
    outputs = model(states)
    # Compute loss
    loss = criterion(outputs, inputs)
    # Backward pass
    optimizer.zero_grad()
    # Compute gradients
    loss.backward()
    # Update weights
    optimizer.step()
    # Store the loss value for plotting
    loss_values.append(loss.item())
  
  # Print loss
  print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
  
  # Check GPU memory
  if device.type == 'cuda':
    allocated_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_memory = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"GPU memory allocated: {allocated_memory:.2f} MB")
    print(f"GPU memory cached:    {reserved_memory:.2f} MB")

# Save model
torch.save(model.state_dict(), 'model.pth')

import matplotlib.pyplot as plt
import numpy as np

np.save('loss_values_6l.npy', loss_values)

# Plot the loss values
plt.plot(loss_values)
plt.grid(True, which='both')
plt.yscale('log')
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')
plt.show()