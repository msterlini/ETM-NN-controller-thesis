import numpy as np
import warnings

class System():
  def __init__(self):

    # Ignore useless warnings from torch
    warnings.filterwarnings("ignore", category=UserWarning)

    ## ========== SYSTEM DEFINITION ==========

    # State of the system variable 
    self.state = None

    # Constants
    self.g = 9.81
    self.m = 0.15
    self.l = 0.5
    self.mu = 0.05
    self.dt = 0.02
    self.max_torque = 5
    self.max_speed = 8.0

    # Fundamental dimensions
    self.nx = 3
    self.nu = 1
    
    # State matrices for the linearized system: x^+ = A*x + B*u
    self.A = np.array([
        [1,                       self.dt,                                0],
        [self.g*self.dt/self.l,   1-self.mu*self.dt/(self.m*self.l**2),   0],
        [1,                       0,                                      1]
    ])
    self.B = np.array([
        [0],
        [self.dt * self.max_torque/(self.m*self.l**2)],
        [0]
    ])

  ## ========== MODULES DEFINITION ==========
  
  # Function to compute the state evolution of the system
  def step(self, u):
    return self.A @ self.state + self.B * u