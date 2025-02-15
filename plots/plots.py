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

colors = ['orange', 'red', 'blue']
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

colors = ['yellow', 'orange', 'red', 'blue']
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

  