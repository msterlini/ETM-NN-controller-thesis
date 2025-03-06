# Thesis Code Repository

This repository contains the implementation of the algorithms and simulations used in my thesis: **"Event-Triggered Methods for Reduced Function Evaluation in Neural Network Controllers"**

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.8+
- NumPy
- SciPy
- CVXPY (with **Mosek** as the solver)
- Stable-Baselines3
- Matplotlib
- Torch

## Repository Structure

```
📂 thesis-code
│-- LMI_results/           # LMI results used in the paper "Layer-wise dynamic event-triggered neural network control for discrete-time nonlinear systems"
│-- auxiliary_code/        # Code used to plot ellipsoids
│-- bilinear_results/      # Last results on bilinear treatment of inclusion conditions
│-- deep_learning/         # Deep learning code to train NN
│-- models/                # NN templates
│-- plots/                 # Code for plot generation
│-- reinforcement_learning/# Reinforcement learning code to train NN
│-- weights/               # Final weights
│-- LMI.py                 # Main LMI execution script
│-- config.py              # Select LMI configuration
│-- system.py              # System under examination
```

## Contact

For questions, feel free to reach out:

- Email: [marcosterlini1@gmail.com](mailto\:marcosterlini1@gmail.com)
