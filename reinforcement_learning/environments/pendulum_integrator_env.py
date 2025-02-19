import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from system import System
import os

class NonLinPendulum_env(gym.Env):
  
  def __init__(self, W, b):
    super(NonLinPendulum_env, self).__init__()

    # Initialize the weights and biases
    self.W = W
    self.b = b

    # Initialize the system to get the parameters
    self.system = System(W, b, [], 0.0, 0.0)

    self.dt = self.system.dt
    self.g = self.system.g
    self.m = self.system.m
    self.l = self.system.l
    self.max_speed = self.system.max_speed
    self.max_torque = self.system.max_torque
    self.nx = self.system.nx
    self.nu = self.system.nu

    # Dynamics matrices
    self.A = self.system.A
    self.B = self.system.B
    self.C = self.system.C
    self.D = self.system.D

    # Reference initialization
    self.ref_bound = 0.5
    self.ref = 0.0

    # Equilibria set
    self.equilibria_set = []
    self.xstar = self.system.xstar

    # P matrix initialization for lypaunov function used for cost
    self.P = np.eye(self.nx)*0
    self.old_P = np.eye(self.nx)*0
    
    # Action space definition
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    
    # Observation space definition
    self.lim_state = np.array([np.pi, self.max_speed, np.inf], dtype=np.float32)
    self.observation_space = spaces.Box(low=-self.lim_state, high=self.lim_state, shape=(3,), dtype=np.float32)
    
    # Time variable
    self.time = 0
    # End of episode
    self.episode_end = 200

    # Empty state initialization
    self.state = None

    self.ROA_reward = 0.0

    self.cumulative_reward = 0

  # Setter methods
  def set_P(self, P, old_P):
    self.P = P
    self.old_P = old_P
  
  def set_ROA_reward(self, ROA_reward):
    self.ROA_reward = ROA_reward

  def get_ROA_reward(self):
    return self.ROA_reward

  def set_equilibria_set(self, equilibria_set):
    self.equilibria_set = equilibria_set
  
  # Getter methods
  def get_P(self):
    return self.P
  
  def get_ref_bound(self):
    return self.ref_bound

  def step(self, action):

    th, thdot, eta = self.state
    
    action = np.clip(action, -1.0, 1.0) * self.max_torque

    # Theta normalization
    th = self.state[0] 
    th = (th + np.pi) % (2*np.pi) - np.pi

    # State update
    state = np.squeeze(self.state)

    # Nonlinear dynamics, the dimension cast are necessary to keep it a vector of dimension (3,)
    new_state = self.A @ state + (self.B * action).reshape(3,) + (self.C * (np.sin(th) - th)).reshape(3,) + (self.D * self.ref).reshape(3,)

    # Update the state
    self.state = np.squeeze(np.array([new_state.astype(np.float32)]))

    # Flag to determine if the cost is computed using the LMI or the standard quadratic cost
    LMI_cost = True

    if LMI_cost:
      # equilibrium point
      xstar = np.squeeze(self.xstar)
      
      # The cost is set equal to the increment of the Lyapunov function.
      # Evolutions that increase the Lyapunov function are penalized, while evolutions that decrease the Lyapunov function are rewarded.
      cost = (new_state - xstar).T @ self.P @ (new_state - xstar) - (state - xstar).T @ self.P @ (state - xstar)

      reward = -float(cost) + self.ROA_reward
      
    else: 
      # Standard quadratic cost
      state_cost = (th**2 + 0.1*thdot**2 +  0.001*(eta**2) + 0.001*(action**2))[0]
      # Reward for staying alive
      stay_alive_reward = 1.0
      
      # Cost function
      reward = float(-state_cost + stay_alive_reward)

    # Flags required for the environment
    truncated = False
    terminated = False
    
    # Check if the episode is terminated
    if self.time >= 200 - 1:
      truncated = True
      
    # A increasing weight is added to the state to check if the state is within the limits, the more the length of the episode, the more the weight
    state_to_check = (1/self.episode_end * self.time + 1) * self.state
    
    # Check if the modified state is within the limits
    if not self.observation_space.contains(state_to_check):
      terminated = True
      
    # Update the time
    self.time += 1

    self.cumulative_reward += reward

    # Return the observation, the cost, the termination flag, the truncated flag and the info
    return self.get_obs(), reward, terminated, truncated, {}
  
  # Reset the environment
  def reset(self, seed=None):
    in_ellip = False

    while not in_ellip:
      # Random initialization of the state
      th = np.float32(np.random.uniform(low=-self.lim_state[0], high=self.lim_state[0]))
      thdot = np.float32(np.random.uniform(low=-self.lim_state[1], high=self.lim_state[1]))
      int = np.float32(0.0)
      # Update the state
      state = np.squeeze(np.array([th, thdot, int]))
      
      if (state.T @ self.P @ state <= 1.5):
        in_ellip = True
        self.state = state
    
    # Resets the time
    self.time = 0

    # Take a random tuple from self.equilibria_set
    idx = np.random.randint(len(self.equilibria_set))
    self.ref, self.xstar = self.equilibria_set[idx]
    # Save the cumulative ROA of the episode

    with open('cumulative_reward.csv', 'a') as f:
      f.write(f"{self.cumulative_reward}\n")
    self.cumulative_reward = 0

    # Return the observation
    return (self.get_obs(), {})
  
  # Get the observation
  def get_obs(self):
    return self.state

# Check the environment
if __name__ == "__main__":
  env = NonLinPendulum_env()
  check_env(env)