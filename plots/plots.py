from auxiliary_code.ellipsoids import ellipsoid_plot_3D
from auxiliary_code.ellipsoids import ellipsoid_plot_2D_projections
import matplotlib.pyplot as plt
import numpy as np
import os


result_paths = ['results_2', 'results_3', 'results_6']
P_mats = []
for path in result_paths:
  files = sorted(os.listdir(os.path.abspath(__file__ + "/../" + path)))
  for f in files:
    if f.startswith('P') and f.endswith('.npy'):
      P_mats.append(np.load(os.path.abspath(__file__ + "/../" + path + "/" + f)))

colors = ['blue', 'orange', 'red']
layers = ['2', '3', '6']

for i in range(len(result_paths)):
  if i == 0:
    fig, ax =  ellipsoid_plot_3D(P_mats[i], False, color=colors[i], legend=layers[i] + r' layers approximation $\mathcal{E}(P, x_*)$')
    ellipsoid_plot_2D_projections(P_mats[i], 'xy', offset=-4, ax=ax, color=colors[i], legend=None)
    ellipsoid_plot_2D_projections(P_mats[i], 'xz', offset=3, ax=ax, color=colors[i], legend=None)
    ellipsoid_plot_2D_projections(P_mats[i], 'yz', offset=-19, ax=ax, color=colors[i], legend=None)
  else:
    ellipsoid_plot_3D(P_mats[i], True, ax=ax, color=colors[i], legend=layers[i] + r' layers approximation $\mathcal{E}(P, x_*)$')
    ellipsoid_plot_2D_projections(P_mats[i], 'xy', offset=-4, ax=ax, color=colors[i], legend=None)
    ellipsoid_plot_2D_projections(P_mats[i], 'xz', offset=3, ax=ax, color=colors[i], legend=None)
    ellipsoid_plot_2D_projections(P_mats[i], 'yz', offset=-19, ax=ax, color=colors[i], legend=None)
ax.legend(fontsize=14)
plt.show()

  
result_paths = ['results_3_rl', 'results_2', 'results_3', 'results_6']
P_mats = []
for path in result_paths:
  files = sorted(os.listdir(os.path.abspath(__file__ + "/../" + path)))
  for f in files:
    if f.startswith('P') and f.endswith('.npy'):
      P_mats.append(np.load(os.path.abspath(__file__ + "/../" + path + "/" + f)))

colors = ['green', 'blue', 'orange', 'red']
layers = ['3', '2', '3', '6']

for i in range(len(result_paths)):
  if i == 0:
    fig, ax =  ellipsoid_plot_3D(P_mats[i], False, color=colors[i], legend=layers[i] + r' layers approximation $\mathcal{E}(P, x_*)$ after RL training')
    ellipsoid_plot_2D_projections(P_mats[i], 'xy', offset=-7, ax=ax, color=colors[i], legend=None)
    ellipsoid_plot_2D_projections(P_mats[i], 'xz', offset=6, ax=ax, color=colors[i], legend=None)
    ellipsoid_plot_2D_projections(P_mats[i], 'yz', offset=-32, ax=ax, color=colors[i], legend=None)
  else:
    ellipsoid_plot_3D(P_mats[i], True, ax=ax, color=colors[i], legend=layers[i] + r' layers approximation $\mathcal{E}(P, x_*)$')
    ellipsoid_plot_2D_projections(P_mats[i], 'xy', offset=-7, ax=ax, color=colors[i], legend=None)
    ellipsoid_plot_2D_projections(P_mats[i], 'xz', offset=6, ax=ax, color=colors[i], legend=None)
    ellipsoid_plot_2D_projections(P_mats[i], 'yz', offset=-32, ax=ax, color=colors[i], legend=None)
ax.legend(fontsize=14, loc='upper left')
plt.show()

loss_values_paths = ['loss_values_2l.npy', 'loss_values_3l.npy', 'loss_values_6l.npy']  

loss_values = [np.load(os.path.abspath(__file__ + "/../" + path)) for path in loss_values_paths]
# Create a figure and axis
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Normal scale plot
ax[0].plot(loss_values[0], label="2 layer loss", color='blue')
ax[0].plot(loss_values[1], label="3 layer loss", color='red')
ax[0].plot(loss_values[2], label="6 layer loss", color='green')
ax[0].set_title('Loss Trend (Normal Scale)', fontsize=18)
ax[0].set_xlabel('Epochs', fontsize=16)
ax[0].set_ylabel('Loss', fontsize=16)
ax[0].grid(True)

# Logarithmic scale plot
ax[1].plot(loss_values[0], label="2 layer loss", color='blue')
ax[1].plot(loss_values[1], label="3 layer loss", color='red')
ax[1].plot(loss_values[2], label="6 layer loss", color='green')
ax[1].set_title('Loss Trend (Logarithmic Scale)', fontsize=18)
ax[1].set_xlabel('Epochs', fontsize=16)
ax[1].set_ylabel('Log(Loss)', fontsize=16)
ax[1].set_yscale('log')  # Set the y-axis to log scale
ax[1].grid(True)

# Show the plot
plt.tight_layout()
ax[0].legend(fontsize=16, loc='upper left')
ax[1].legend(fontsize=16, loc='upper left')
plt.show()