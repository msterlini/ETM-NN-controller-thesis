from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
from reinforcement_learning.environments.pendulum_integrator_env import NonLinPendulum_env
from models.NN3l import NeuralNet
from LMI import LMI
from system import System

# User warnings filter
warnings.filterwarnings("ignore", category=UserWarning, module='stable_baselines3')
warnings.filterwarnings("ignore", category=UserWarning, module='gymnasium')

# Device declaration to exploit GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

folder = '../deep_learning/3_layers/weights'

## ======== WEIGHTS AND BIASES IMPORT ========
files = sorted(os.listdir(os.path.abspath(__file__ + "/../" + folder)))
W = []
b = []
for f in files:
  if f.startswith('W') and f.endswith('.csv'):
    W.append(np.loadtxt(os.path.abspath(__file__ + "/../" + folder + '/' + f), delimiter=','))
  elif f.startswith('b') and f.endswith('.csv'):
    b.append(np.loadtxt(os.path.abspath(__file__ + "/../" + folder + '/' + f), delimiter=','))

# Weights and biases reshaping
W[-1] = W[-1].reshape((1, len(W[-1])))

# Custom environment declaration
env = NonLinPendulum_env(W, b)
ref_bound = env.get_ref_bound()

def create_batch_equilibrium(W, b, n):
  equilibria = []

  for i in range(n):
    ref  = np.random.uniform(-ref_bound, ref_bound)
    s = System(W, b, [], ref, 0.0)
    equilibria.append((ref, s.xstar))
    print(f'\rEquilibrium {i+1}/{n} created', end='')
  print('')
  return equilibria

n_equilibria = 10
equilibria_set = create_batch_equilibrium(W, b, n_equilibria)
env.set_equilibria_set(equilibria_set)
  
lmi = LMI(W, b)
lmi.solve(0.1)
lmi.save_results('LMI_results')

# Model network architecture extraction
neural_net = NeuralNet(env.nx)
net_arch = neural_net.arch

# Policy declaration to impose the structure and activation function type
policy_kwargs = dict(activation_fn=torch.nn.Hardtanh, net_arch=net_arch)

# Model declaration
print(device)
model_rl = PPO(
  "MlpPolicy", 
  env, 
  policy_kwargs=policy_kwargs, 
  verbose=1,
  device=device
)

# Custom callback class to handle different events during training:
# - Training start: load the current best weights from the LMI optimization
# - Rollout start: reset the model to the last best weights if the current model is not improving or is infeasible
# - Rollout end: solve the LMI optimization problem to find the new best weights
# - Step: not used but forced to return True to avoid errors

# The callback also saves the best model and the current model at the end of each rollout
class CustomCallback(BaseCallback):
  def __init__(self, verbose = 0):
    super().__init__(verbose)

    self.nlayers = env.system.nlayers

    # Path to import the best weights, LQR is set to True to load the LQR weights
    self.LQR = False
    if self.LQR:
      self.param_path = os.path.abspath(__file__ + '/../policy.pth')
    else:
      self.param_path = os.path.abspath(__file__ + '/../best_policy.pth')

    # Variables to store the best ROA area during training
    self.best_ROA = 0.0
    # Variable to store current rollout attempt without improvement
    self.n_rollout_no_improvement = 0
    # User imposed limit to the number of rollouts without improvement
    self.n_rollout_limit = 10
    # Flag variable to reset the model to the last best weights
    self.reset = 0
  
  # Function to extract the weights from the model, it is important to clone the data otherwise only a deep copy of the reference is created
  def get_weights(self):
    W = []
    b = []
    
    # Hidden layers extraction
    for i in range(self.nlayers - 1):
      weight = self.model.policy.mlp_extractor.policy_net[i*2].weight.data.clone().detach().to(device).numpy()
      bias = self.model.policy.mlp_extractor.policy_net[i*2].bias.data.clone().detach().to(device).numpy()
      W.append(weight)
      b.append(bias)
    
    # Output layer extraction
    weight = self.model.policy.action_net.weight.data.clone().detach().to(device).numpy()
    bias = self.model.policy.action_net.bias.data.clone().detach().to(device).numpy()
    W.append(weight)
    b.append(bias)
    return W, b

  # Callback called on the training start
  def _on_training_start(self):
    
    # Load the best weights
    state_dict = torch.load(self.param_path, map_location=torch.device(device), weights_only=True)    

    # The flag LQR is necessary since the two state_dict have different names for their elements
    if self.LQR: 
      
      # Hidden layers
      for i in range(self.nlayers - 1):
        id = 'l' + str(i+1) 
        self.model.policy.mlp_extractor.policy_net[i*2].weight = nn.Parameter(state_dict[id + '.weight'].clone().detach().requires_grad_(True))
        self.model.policy.mlp_extractor.policy_net[i*2].bias = nn.Parameter(state_dict[id + '.bias'].clone().detach().requires_grad_(True))
      
      # Output layer
      id = 'l' + str(self.nlayers) 
      self.model.policy.action_net.weight = nn.Parameter(state_dict[id + '.weight'].clone().detach().requires_grad_(True))
      self.model.policy.action_net.bias = nn.Parameter(state_dict[id + '.bias'].clone().detach().requires_grad_(True))

    else:
      for i in range(self.nlayers - 1):
        id = str(i*2) 
        self.model.policy.mlp_extractor.policy_net[i*2].weight = nn.Parameter(state_dict['mlp_extractor.policy_net.' + id + '.weight'].clone().detach().requires_grad_(True))
        self.model.policy.mlp_extractor.policy_net[i*2].bias = nn.Parameter(state_dict['mlp_extractor.policy_net.' + id + '.bias'].clone().detach().requires_grad_(True))
      
      # Output layer
      self.model.policy.action_net.weight = nn.Parameter(state_dict['action_net.weight'].clone().detach().requires_grad_(True))
      self.model.policy.action_net.bias = nn.Parameter(state_dict['action_net.bias'].clone().detach().requires_grad_(True))

    # Reinitialize the optimizer, otherwise the model will keep the previous optimizer state and will not improve
    optim_class = type(self.model.policy.optimizer)
    optim_param = self.model.policy.optimizer.defaults
    self.model.policy.optimizer = optim_class(self.model.policy.parameters(), **optim_param)

    # Just at the beginning of the training the P matrix is the result of the last standalone LMI problem solution
    try:
      P = np.load('LMI_results/P.npy')
    except:
      P = np.eye(env.nx)
    self.model.get_env().env_method('set_P', P, P)


  def _on_step(self):
    return True

  # Callback called on the rollout end after the optimization step of sb3 that updates the policy weights. Will be used to solve the LMI on the new weights and update the best weights accordingly.
  # If the policy update is infeasible the reset flag will handle the weights reset at the beginning of the next rollout
  def _on_rollout_end(self):
    
    # Weights and biases extraction
    W, b = self.get_weights()
    # LMI class initialization

    lmi = LMI(W, b)  
    # LMI problem solution, alpha value empirically set to 0.1 to have a feasible solution that is not too conservative. The alpha value search will not be performed mid-training to save processing time. Meaning that potentially a slightly better ROA solution can be found by performing it after the training.
    
    print('Solving LMI..')
    P = lmi.solve(0.1)

    
    # Following the logic implemented in the LMI P is None for a infeasible solution.
    if P is not None:

      create_batch_equilibrium(W, b, n_equilibria)
      self.model.get_env().env_method('set_equilibria_set', equilibria_set)
      # ROA volume calculation
      volume = 4/3 * np.pi/np.sqrt(np.linalg.det(P))
      # Saving current policy if feasible
      self.model.save('rollout_model.zip')

      # If the result is the best so far, update the best weights and reset the no improvement counter. Also save the best model.
      if volume > self.best_ROA:
        # Reset no improvement counter
        self.n_rollout_no_improvement = 0

        # P matrix update in the environment
        old_P = self.model.get_env().env_method('get_P')
        self.model.get_env().env_method('set_P', P, old_P)

        # Best ROA update
        self.best_ROA = volume
        
        # Best weights update
        self.best_W = W
        self.best_b = b

        # Reward update in the environment, the reward is the difference between the current area and the best area, in this case I prefer to give directly the new ROA as a reward to further encourage the policy to find improvements
        # self.model.get_env().env_method('set_ROA_reward', self.best_ROA/200.0)
        self.model.get_env().env_method('set_ROA_reward', (self.best_ROA)/200.0)

        # Model save
        self.model.save('best_rollout_model.zip')
        print(f'New best model saved, ROA: {self.best_ROA}')
      # In the case of feasible policy update but not optimal
      else:
        # Increment no improvement counter
        self.n_rollout_no_improvement += 1
        print(f"Feasible increment, but not better than best model. Best ROA: {self.best_ROA}")
        print(f'Difference of volume: {volume - self.best_ROA}')
        print(f"Keeping current P")
        print(f"Current rollout attempt: {self.n_rollout_no_improvement}") # Reward update in the environment, the reward is the difference between the current volum and the best volume so it is a cost in this case
        # self.model.get_env().env_method('set_ROA_reward', (volume - self.best_ROA)/200.0)
        self.model.get_env().env_method('set_ROA_reward', (self.best_ROA)/200.0)

        # If the no improvement counter reaches the limit, reset the model to the last best weights
        if self.n_rollout_no_improvement >= self.n_rollout_limit:
          print(f"Resetting to last best model")
          self.reset = 1
    
    # In the case of infeasible policy update reset the model to the last best weights
    else:
      print(f'Infeasible increment, keeping current P')
      print(f"Resetting to last best model")
      self.reset = 1

  # Callback called on the rollout start, if the reset flag is set the model will be reset to the last best weights
  def _on_rollout_start(self):
    if self.reset:
      self.reset = 0
      self.n_rollout_no_improvement = 0
      
      # Hidden layers
      for i in range(self.nlayers - 1):
        self.model.policy.mlp_extractor.policy_net[i*2].weight = nn.Parameter(torch.tensor(self.best_W[i], requires_grad=True))
        self.model.policy.mlp_extractor.policy_net[i*2].bias = nn.Parameter(torch.tensor(self.best_b[i], requires_grad=True))

      # Output layer
      self.model.policy.action_net.weight = nn.Parameter(torch.tensor(self.best_W[-1], requires_grad=True))
      self.model.policy.action_net.bias = nn.Parameter(torch.tensor(self.best_b[-1], requires_grad=True))
    
      # Reinitialize the optimizer, otherwise the model will keep the previous optimizer state and will not improve
      optim_class = type(self.model.policy.optimizer)
      optim_param = self.model.policy.optimizer.defaults
      self.model.policy.optimizer = optim_class(self.model.policy.parameters(), **optim_param)
      self.reset = 0

# Custom evaluation callback to handle the evaluation of the model during training, it automatically saves the best model based on the evaluation score
CustomEvalCallback = EvalCallback(env, best_model_save_path='.', log_path='./logs', eval_freq=1000, deterministic=True, render=False, verbose=0)

# Callback list to handle the different custom callbacks
# callback = CallbackList([CustomCallback(), CustomEvalCallback])
# Callback to not take into account the evaluation callback
callback = CustomCallback()

# Model training
model_rl.learn(total_timesteps=1000000, callback=callback, progress_bar=True)

## Testing the best model after training

# Empty lists to store the states and inputs of the episodes
states = []
inputs = []
episode_state = []
episode_input = []

# Variables to store the total number of episodes and the number of converging episodes
n_tot = 0
n_converging = 0

# Environment reset
vec_env = model_rl.get_env()
obs = vec_env.reset()

# Loop to run the episodes
for i in range(10000):
  action, _states = model_rl.predict(obs, deterministic=True)
  obs, rewards, done, info = vec_env.step(action)
  episode_state.append(obs)
  episode_input.append(action)

  # If the episode is done, store the episode and reset the episode lists
  if done:
    n_tot += 1
    states.append(episode_state)
    inputs.append(episode_input)
    episode_state = []
    episode_input = []

## Result plotting    
for episode in states:
  if len(episode) > 50:
    n_converging += 1
    episode = np.squeeze(np.array(episode))
    plt.plot(episode[:,0], episode[:,1])
plt.show()

for episode in inputs:
  if len(episode) > 50:
    episode = np.squeeze(np.array(episode))
    plt.plot(episode)
plt.show()

print(f'Converging episodes: {n_converging}/{n_tot}')
