from system import System
import config as conf
import numpy as np
import cvxpy as cp
import warnings
import os

class LMI():
  def __init__(self, W, b):
    
    # Declare system to import values
    self.system   = System(W, b, [], 0.0, 0.0)
    self.nx       = self.system.nx
    self.nu       = self.system.nu
    self.nq       = self.system.nq
    self.nphi     = self.system.nphi
    self.neurons  = self.system.neurons
    self.nlayers  = self.system.nlayers
    self.wstar    = self.system.wstar
    self.ustar    = self.system.ustar
    self.bound    = self.system.bound

    # Flag variables to determine which kind of LMI has to be solved
    self.optim        = conf.optim
    
    # Sign definition of Delta V parameter
    self.m_thres = 1e-6

    # Parameters definition
    self.alpha = cp.Parameter()

    # Auxiliary matrices
    self.Abar = self.system.Abar
    self.Bbar = self.system.Bbar
    self.C    = self.system.C
    self.R    = self.system.R
    self.Nvx  = self.system.Nvx
    
    # Projection matrices
    self.Pi_nu = self.system.Pi_nu
    self.Pi_s  = self.system.Pi_s

    # Quadratic abstraction for nonlinearity
    self.Phi_abstraction = self.system.Phi_abstraction

    # Projection matrices
    self.projection_matrices = self.system.projection_matrices

    # Function that handles all Variables declarations
    self.init_variables()

    # Function that handles all Constraints declarations
    self.init_constraints()

    # Function that handles final problem definition
    self.create_problem()
  
  # Function that handles all Variables declarations
  def init_variables(self):

    # P matrix for Lyapunov function
    self.P = cp.Variable((self.nx, self.nx), symmetric=True)

    # ETM Variables
    S_val         = cp.Variable(self.nphi)
    self.S        = cp.diag(S_val)
    self.S_layers = []
    start = 0
    for i in range(self.nlayers):
      end = start + self.neurons[i]
      self.S_layers.append(self.S[start:end, start:end])
      start = end

    self.G        = cp.Variable((self.nphi, self.nx))
    self.G_layers = []
    start = 0
    for i in range(self.nlayers):
      end = start + self.neurons[i]
      self.G_layers.append(self.G[start:end, :])
      start = end

    # Finsler multipliers, structured to reduce computational burden and different for each layer
    self.finsler_multipliers = []
    for i in range(self.nlayers):
      N1  = cp.Variable((self.nx, self.nphi))
      N2  = cp.Variable((self.nphi, self.nphi), symmetric=True)
      N3  = cp.diag(cp.Variable(self.nphi))
      N   = cp.vstack([N1, N2, N3])
      self.finsler_multipliers.append(N)
    
    # Finlser multipliers for new structure
    self.new_finsler_multipliers = []
    for i in range(self.nlayers):
      size_tot  = self.nx + 2 * self.nphi
      size      = self.nx + 2 * self.neurons[i]
      Theta11   = cp.Variable((size_tot, self.nx))
      Theta13   = cp.Variable((size_tot, self.neurons[i]))
      Theta21   = cp.Variable((size, self.nx))
      Theta23   = cp.Variable((size, self.neurons[i]))

      if self.nx > self.neurons[i]:
        n_id = self.nx // self.neurons[i]
        n_zeros = self.nx % self.neurons[i]
        block1 = cp.vstack([np.eye(self.neurons[i]) for _ in range(n_id)])
        if n_zeros != 0:
          block1 = cp.vstack([block1, np.zeros((n_zeros, self.neurons[i]))])
      elif self.nx < self.neurons[i]:
        n_id = self.neurons[i] // self.nx
        n_zeros = self.neurons[i] % self.nx
        block1 = cp.hstack([np.eye(self.nx) for _ in range(n_id)])
        if n_zeros != 0:
          block1 = cp.hstack([block1, np.zeros((self.nx, n_zeros))])
      else:
        block1 = np.eye(self.nx)

      n_id = self.nphi // self.neurons[i]
      n_zeros = self.nphi % self.neurons[i]
      block2 = cp.vstack([np.eye(self.neurons[i]) for _ in range(n_id)])
      if n_zeros != 0:
        block2 = cp.vstack([block2, np.zeros((n_zeros, self.neurons[i]))])

      Theta12 = self.alpha * cp.vstack([block1, block2, block2])
      Theta22 = self.alpha * cp.vstack([block1, np.eye(self.neurons[i]), np.eye(self.neurons[i])])

      Theta1    = cp.hstack([Theta11, Theta12, Theta13])
      Theta2    = cp.hstack([Theta21, Theta22, Theta23])
      self.new_finsler_multipliers.append([Theta1, Theta2])

    self.F1         = cp.Variable((self.nx, self.nphi))
    self.K          = cp.Variable(self.nphi)
    self.F3         = cp.Variable(self.nphi)
    self.F1_layers  = []
    self.K_layers   = []
    self.F3_layers  = []
    start = 0
    for i in range(self.nlayers):
      end = start + self.neurons[i]
      self.F1_layers.append(self.F1[:, start:end])
      self.K_layers.append(self.K[start:end])
      self.F3_layers.append(self.F3[start:end])
      start = end
      
    # New ETM matrices
    self.bigX_matrices = []
    for i in range(self.nlayers):
      size = self.nx + 2 * self.neurons[i]
      bigX = cp.Variable((size, size))
      self.bigX_matrices.append(bigX)
    
    # Variable for Sigma
    eps = cp.Variable(self.nx + self.nphi + self.nq)
    self.eps = cp.diag(eps)
    
    # ETM minimization variables
    self.betas = []
    for i in range(self.nlayers):
      beta = cp.Variable(nonneg=True)
      self.betas.append(beta)
  
  # Function that handles all Constraints declarations
  def init_constraints(self):

    # Delta V matrix formulation with non-linearity sector condition, beign positive definite in -pi, pi it's added as a positive term
    self.M = cp.bmat([
      [self.Abar.T],
      [self.Bbar.T],
      [self.C.T]
    ]) @ self.P @ cp.bmat([[self.Abar, self.Bbar, self.C]]) - cp.bmat([
      [self.P,                          np.zeros((self.nx, self.nphi)),   np.zeros((self.nx, self.nq))],
      [np.zeros((self.nphi, self.nx)),  np.zeros((self.nphi, self.nphi)), np.zeros((self.nphi, self.nq))],
      [np.zeros((self.nq, self.nx)),    np.zeros((self.nq, self.nphi)),   np.zeros((self.nq, self.nq))]
    ]) + self.Pi_s.T @ self.Phi_abstraction @ self.Pi_s

    # ETM constraints
    # Structure of sector condition to add to finsler constraint
    # self.Omegas = []
    # for i in range(self.nlayers):
    #   Omega = cp.bmat([
    #     [np.zeros((self.nx, self.nx)), np.zeros((self.nx, self.neurons[i])), np.zeros((self.nx, self.neurons[i]))],
    #     [self.Z_layers[i], self.T_layers[i], -self.T_layers[i]],
    #     [np.zeros((self.neurons[i], self.nx)), np.zeros((self.neurons[i], self.neurons[i])), np.zeros((self.neurons[i], self.neurons[i]))]
    #   ])
    #   self.Omegas.append(Omega)

    # Addition of sector conditions to Delta V matrix
    for i in range(self.nlayers):
      self.M += -self.Pi_nu.T @ (self.projection_matrices[i].T @ (self.bigX_matrices[i] + self.bigX_matrices[i].T) @ self.projection_matrices[i]) @ self.Pi_nu

    # Definition of Ker([x, psi, nu]) to add in the finsler constraints
    self.hconstr = cp.hstack([self.R @ self.Nvx, np.eye(self.R.shape[0]) - self.R, -np.eye(self.nphi)])

    # Finsler constraints for each layer
    self.finsler_constraints = []
    for i in range(self.nlayers):
      # finsler = self.projection_matrices[i].T @ (self.bigX_matrices[i] - self.Omegas[i] + self.bigX_matrices[i].T - self.Omegas[i].T) @ self.projection_matrices[i] + self.finsler_multipliers[i] @ self.hconstr + self.hconstr.T @ self.finsler_multipliers[i].T
      # self.finsler_constraints.append(finsler)
      finsler1 = self.projection_matrices[i].T @ (self.bigX_matrices[i] + self.bigX_matrices[i].T) @ self.projection_matrices[i] + self.finsler_multipliers[i] @ self.hconstr + self.hconstr.T @ self.finsler_multipliers[i].T + self.new_finsler_multipliers[i][0] @ self.projection_matrices[i] + self.projection_matrices[i].T @ self.new_finsler_multipliers[i][0].T
      finsler2 = -self.new_finsler_multipliers[i][0] + self.projection_matrices[i].T @ self.new_finsler_multipliers[i][1].T
      upsilon = cp.bmat([
        [np.zeros((self.nx, self.nx)), self.G_layers[i].T, np.zeros((self.nx, self.neurons[i]))],
        [self.G_layers[i], 2 * self.S_layers[i], -np.eye(self.neurons[i])],
        [np.zeros((self.neurons[i], self.nx)), -np.eye(self.neurons[i]), np.zeros((self.neurons[i], self.neurons[i]))]
      ])
      finsler3 = -upsilon - self.new_finsler_multipliers[i][1] - self.new_finsler_multipliers[i][1].T
      finsler = cp.bmat([
        [finsler1, finsler2],
        [finsler2.T, finsler3]
      ])
      self.finsler_constraints.append(finsler)
   
    # Constraint definition 
    self.constraints = [self.P >> 0]
    self.constraints += [self.S >> 0]
    # self.constraints += [self.S << 1e5 * np.eye(self.S.shape[0])]
    self.constraints += [self.M << -self.m_thres * np.eye(self.M.shape[0])]
    # self.constraints += [self.eps >> 0]
    # self.constraints += [self.M + self.eps >> 0]
    for constraint in self.finsler_constraints:
      self.constraints += [constraint << 0]

    # Minimization constraints of X_i for each layer
    # if self.optim:
    #   for i in range(self.nlayers):
    #     id = np.eye(self.nx + 2 * self.neurons[i])
    #     mat = cp.bmat([
    #       [-self.betas[i] * id, self.bigX_matrices[i]],
    #       [self.bigX_matrices[i].T, -id]
    #     ])
    #     self.constraints += [mat << 0]
    
    # Ellipsoid conditions for activation functions
    for i in range(self.nlayers):
      for k in range(self.neurons[i]):
        G_el  = cp.reshape(self.G_layers[i][k, :], (1, self.nx))
        F1_el = cp.reshape(self.F1_layers[i][:, k], (self.nx, 1))
        K_el  = cp.reshape(self.K_layers[i][k], (1,1))
        F3_el = cp.reshape(self.F3_layers[i][k], (1,1))
        v_el  = cp.reshape(np.min([np.abs(-self.bound - self.wstar[i][k][0]), np.abs(self.bound - self.wstar[i][k][0])], axis=0), (1,1))
        ellip = cp.bmat([
          [self.P, G_el.T + v_el * F1_el, -F1_el],
          [G_el + v_el * F1_el.T, v_el * (K_el + K_el.T), -K_el + v_el * F3_el.T],
          [-F1_el.T, -K_el.T + v_el * F3_el, 1 - F3_el - F3_el.T]
        ])
        self.constraints += [ellip >> 0]
    
  # Function that handles final problem definition
  def create_problem(self):

    # Objective function defined as the sum of the trace of P, eps and the sum of all alphax variables
    # if self.optim:
    #   obj = cp.trace(self.P) + cp.trace(self.eps)
    #   for i in range(self.nlayers):
    #     obj += self.betas[i]
    # else:
      # obj = cp.trace(self.P) + cp.trace(self.eps)
    obj = cp.trace(self.P)

    self.objective = cp.Minimize(obj)

    # Problem definition
    self.prob = cp.Problem(self.objective, self.constraints)

    # Warnings disabled only for clearness during debug procedures
    # User warnings filter
    warnings.filterwarnings("ignore", category=UserWarning, module='cvxpy')

  # Function that takes parameter values as input and solves the LMI
  def solve(self, alpha_val, verbose=False): #, search=False):
    # Parameters update
    self.alpha.value = alpha_val

    try:
      self.prob.solve(solver=cp.MOSEK, verbose=verbose)
      # self.prob.solve(solver=cp.SCS, verbose=verbose, max_iters=1000000)
    except cp.error.SolverError:
      return None

    if self.prob.status not in ["optimal", "optimal_inaccurate"]:
      return None
    else:
      print(f"Max eigenvalue of P: {np.max(np.linalg.eigvals(self.P.value))}")
      print(f"Max eigenvalue of M: {np.max(np.linalg.eigvals(self.M.value))}") 
      print(f"Size of ROA: {4/3 * np.pi/np.sqrt(np.linalg.det(self.P.value))}")
      
      # Returns area of ROA if feasible
      return self.P.value
  
  # Function that searches for the optimal alpha value by performing a golden ratio search until a certain numerical accuracy is reached or the limit of iterations is reached 
  def search_alpha(self, feasible_extreme, infeasible_extreme, threshold, verbose=False):

    golden_ratio = (1 + np.sqrt(5)) / 2
    i = 0
    
    # Loop until the difference between the two extremes is smaller than the threshold or the limit of iterations is reached
    while (feasible_extreme - infeasible_extreme > threshold) and i < 1e15:

      i += 1
      alpha1 = feasible_extreme - (feasible_extreme - infeasible_extreme) / golden_ratio
      alpha2 = infeasible_extreme + (feasible_extreme - infeasible_extreme) / golden_ratio
      
      # Solve the LMI for the two alpha values
      Psol = self.solve(alpha1, verbose=False)
      if Psol is None:
        val1 = -1
      else:
        val1 = 4/3 * np.pi / np.sqrt(np.linalg.det(Psol))
      
      Psol = self.solve(alpha2, verbose=False)
      if Psol is None:
        val2 = -1
      else:
        val2 = 4/3 * np.pi / np.sqrt(np.linalg.det(Psol))
        
      # Update the feasible and infeasible extremes
      if val1 > val2:
        feasible_extreme = alpha2
      else:
        infeasible_extreme = alpha1
        
      if verbose:
        if val1 > val2:
          ROA = val1
        else:
          ROA = val2
        print(f"\nIteration number: {i}")
        print(f"==================== \nCurrent ROA value: {ROA}")
        print(f"Current alpha value: {feasible_extreme}\n==================== \n")
    return feasible_extreme

  # Function that saves the variables of interest to use in the simulations
  def save_results(self, path_dir: str):
    if not os.path.exists(path_dir):
      os.makedirs(path_dir)
    np.save(f"{path_dir}/P.npy", self.P.value)
    for id, bigX in enumerate(self.bigX_matrices):
      np.save(f"{path_dir}/bigX{id+1}.npy", bigX.value)
      
# Main loop execution 
if __name__ == "__main__":
  import os

  ## ======== WEIGHTS AND BIASES IMPORT ========

  # # folder = 'deep_learning/2_layers/weights'
  # folder = 'deep_learning/3_layers/weights'
  # # folder = 'weights'

  # files = sorted(os.listdir(os.path.abspath(__file__ + "/../" + folder)))
  # W = []
  # b = []
  # for f in files:
  #   if f.startswith('W') and f.endswith('.csv'):
  #     W.append(np.loadtxt(os.path.abspath(__file__ + "/../" + folder + '/' + f), delimiter=','))
  #   elif f.startswith('b') and f.endswith('.csv'):
  #     b.append(np.loadtxt(os.path.abspath(__file__ + "/../" + folder + '/' + f), delimiter=','))

  # # Weights and biases reshaping
  # W[-1] = W[-1].reshape((1, len(W[-1])))
  
  W = [np.load('deep_learning/K.npy')]
  b = [np.array([np.float32(0)])]

  # Lmi object creation
  lmi = LMI(W, b)

  # Search of alpha value with golden section search
  alpha = lmi.search_alpha(10.0, -10.0, 1e-8, verbose=True)

  # alphas = np.linspace(-1, 1, 10000)
  # for alpha in alphas:
  #   if lmi.solve(alpha, verbose=False):
  #     print('SOLUTION FOUND')
  #     break
  #   else:
  #     print(f'NO SOLUTION FOUND for alpha: {alpha:.6f}')

  # Alpha value import coming from previous simulations
  # alpha = np.load('weights/alpha.npy')

  # LMI solving
  lmi.solve(10, verbose=True)

  # LMI results storage
  # lmi.save_results('new_results')