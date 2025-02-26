from auxiliary_code.ellipsoids import ellipsoid_plot_3D
from auxiliary_code.ellipsoids import ellipsoid_plot_2D_projections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# # result_paths = ['results_2', 'results_3', 'results_6']
# # P_mats = []
# # for path in result_paths:
# #   files = sorted(os.listdir(os.path.abspath(__file__ + "/../" + path)))
#   for f in files:
#     if f.startswith('P') and f.endswith('.npy'):
#       P_mats.append(np.load(os.path.abspath(__file__ + "/../" + path + "/" + f)))

# colors = ['blue', 'orange', 'red']
# layers = ['2', '3', '6']

# for i in range(len(result_paths)):
#   if i == 0:
#     fig, ax =  ellipsoid_plot_3D(P_mats[i], False, color=colors[i], legend=layers[i] + r' layers approximation $\mathcal{E}(P, x_*)$')
#     ellipsoid_plot_2D_projections(P_mats[i], 'xy', offset=-4, ax=ax, color=colors[i], legend=None)
#     ellipsoid_plot_2D_projections(P_mats[i], 'xz', offset=3, ax=ax, color=colors[i], legend=None)
#     ellipsoid_plot_2D_projections(P_mats[i], 'yz', offset=-19, ax=ax, color=colors[i], legend=None)
#   else:
#     ellipsoid_plot_3D(P_mats[i], True, ax=ax, color=colors[i], legend=layers[i] + r' layers approximation $\mathcal{E}(P, x_*)$')
#     ellipsoid_plot_2D_projections(P_mats[i], 'xy', offset=-4, ax=ax, color=colors[i], legend=None)
#     ellipsoid_plot_2D_projections(P_mats[i], 'xz', offset=3, ax=ax, color=colors[i], legend=None)
#     ellipsoid_plot_2D_projections(P_mats[i], 'yz', offset=-19, ax=ax, color=colors[i], legend=None)
# ax.legend(fontsize=14)
# plt.show()

  
# result_paths = ['results_3_rl', 'results_2', 'results_3', 'results_6']
# P_mats = []
# for path in result_paths:
#   files = sorted(os.listdir(os.path.abspath(__file__ + "/../" + path)))
#   for f in files:
#     if f.startswith('P') and f.endswith('.npy'):
#       P_mats.append(np.load(os.path.abspath(__file__ + "/../" + path + "/" + f)))

# colors = ['green', 'blue', 'orange', 'red']
# layers = ['3', '2', '3', '6']

# for i in range(len(result_paths)):
#   if i == 0:
#     fig, ax =  ellipsoid_plot_3D(P_mats[i], False, color=colors[i], legend=layers[i] + r' layers approximation $\mathcal{E}(P, x_*)$ after RL training')
#     ellipsoid_plot_2D_projections(P_mats[i], 'xy', offset=-7, ax=ax, color=colors[i], legend=None)
#     ellipsoid_plot_2D_projections(P_mats[i], 'xz', offset=6, ax=ax, color=colors[i], legend=None)
#     ellipsoid_plot_2D_projections(P_mats[i], 'yz', offset=-32, ax=ax, color=colors[i], legend=None)
#   else:
#     ellipsoid_plot_3D(P_mats[i], True, ax=ax, color=colors[i], legend=layers[i] + r' layers approximation $\mathcal{E}(P, x_*)$')
#     ellipsoid_plot_2D_projections(P_mats[i], 'xy', offset=-7, ax=ax, color=colors[i], legend=None)
#     ellipsoid_plot_2D_projections(P_mats[i], 'xz', offset=6, ax=ax, color=colors[i], legend=None)
#     ellipsoid_plot_2D_projections(P_mats[i], 'yz', offset=-32, ax=ax, color=colors[i], legend=None)
# ax.legend(fontsize=14, loc='upper left')
# plt.show()

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
# Plot with transparency
ax[1].plot(loss_values[0], label="2 layer loss", color='blue', alpha=0.3)
ax[1].plot(loss_values[1], label="3 layer loss", color='red', alpha=0.3)
ax[1].plot(loss_values[2], label="6 layer loss", color='green', alpha=0.3)

# Smooth evolution using a moving average
window_size = 1000
smooth_loss_values = [np.convolve(loss, np.ones(window_size)/window_size, mode='valid') for loss in loss_values]

# Plot the smoothed values
ax[1].plot(smooth_loss_values[0], color='blue')
ax[1].plot(smooth_loss_values[1], color='red')
ax[1].plot(smooth_loss_values[2], color='green')

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


# # Load the rewards from the CSV file
# rewards_df = pd.read_csv(os.path.abspath(__file__ + "/../roa_reward.csv"), header=None, names=['reward'])

# # Extract the rewards values
# rewards = rewards_df['reward'].values

# # Create a new figure for the rewards plot
# fig, ax = plt.subplots(figsize=(10, 5))

# # Plot the rewards with transparency
# ax.plot(rewards, label=r"$V(\mathcal{E}(x, \eta))$", color='black', alpha=0.7)

# # Calculate the cumulative maximum of the rewards
# cumulative_max_rewards = np.maximum.accumulate(rewards)

# # Plot the cumulative maximum rewards
# ax.plot(cumulative_max_rewards, label=r"Best $V(\mathcal{E}(x, \eta))$", color='darkorange')

# # Set the title and labels
# ax.set_title('ROA improvement', fontsize=18)
# ax.set_xlabel('Episodes', fontsize=16)
# ax.set_ylabel(r"$V(\mathcal{E}(x, \eta))$", fontsize=16)
# ax.grid(True)

# # Show the legend
# ax.legend(fontsize=16, loc='lower right')

# # Show the plot
# plt.tight_layout()
# plt.show()

# Load the cumulative rewards from the CSV file
cumulative_rewards_df = pd.read_csv(os.path.abspath(__file__ + "/../cumulative_reward.csv"), header=None, names=['cumulative_reward'])

# Extract the cumulative rewards values and cut to the first 4000 values
cumulative_rewards = cumulative_rewards_df['cumulative_reward'].values[:2000]

# Create a new figure for the cumulative rewards plot
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the cumulative rewards with transparency
ax.plot(cumulative_rewards, label='Cumulative Reward', color='blue', alpha=0.3)

# Smooth evolution using a moving average
window_size = 100
smooth_cumulative_rewards = np.convolve(cumulative_rewards, np.ones(window_size)/window_size, mode='valid')

# Plot the smoothed values
ax.plot(smooth_cumulative_rewards, color='blue')

# Set the title and labels
ax.set_title('Cumulative Rewards Over Time', fontsize=18)
ax.set_xlabel('Episodes', fontsize=16)
ax.set_ylabel('Cumulative Reward', fontsize=16)
ax.grid(True)

# Show the legend
ax.legend(fontsize=16, loc='lower right')

# Show the plot
plt.tight_layout()
plt.show()