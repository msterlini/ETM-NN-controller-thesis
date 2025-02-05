from lin_system import System
import numpy as np
import pandas as pd
import control

# Linear system initialization
s = System()

# Matrix import
A = s.A
B = s.B

# Definition of Q and R matrices
Q = np.array([
  [100, 0, 0],
  [0, 10, 0],
  [0, 0, 1]
])

R = np.array([[1]])

# Computation of the optimal gain matrix K
K, _, _ = control.dlqr(A, B, Q, R)

print(f"K: {K}")

# Eigenvalues of the closed loop system
eigvals = np.linalg.eigvals(A - B @ K)
print(f"Eigvals of closed loop system: {eigvals}")

# Check if the system is stable
stable = all(abs(eig) < 1 for eig in eigvals)
if stable:
  print("System is stable")
  np.save('K.npy', -K)
else:
  print("System is not stable")

## Dataset creation
size_dataset = 1000000

theta_lim = 60 * np.pi / 180

# Mean and standard deviation for the normal distribution of the states
theta_mean = 0
theta_std = theta_lim / 3  # 3-sigma rule for 99.7% coverage
vtheta_mean = 0
vtheta_std = s.max_speed / 3  # 3-sigma rule for 99.7% coverage
int_mean = 0
int_std = 1  # Assuming a standard deviation for integrator state

dataset = []

for _ in range(size_dataset):
  # Creation of random state
  theta = np.clip(np.random.normal(theta_mean, theta_std), -theta_lim, theta_lim)
  vtheta = np.clip(np.random.normal(vtheta_mean, vtheta_std), -s.max_speed, s.max_speed)
  int = np.random.normal(int_mean, int_std)
  state = np.array([[theta], [vtheta], [int]])
  
  # Computation of the control input
  u = -K @ state

  # Application of saturation
  u = np.clip(u, -s.max_torque, s.max_torque)/s.max_torque
  data = np.concatenate([state, u], axis=0)
  dataset.append(data)

# Dataset reshaping and shuffling
dataset = np.array(dataset).reshape(size_dataset, s.nx + s.nu)
np.random.shuffle(dataset)

# Dataset saving
column_labels = ['theta', 'vtheta', 'int', 'u']
df  = pd.DataFrame(dataset, columns=column_labels)
df.to_csv('dataset.csv', index=False)