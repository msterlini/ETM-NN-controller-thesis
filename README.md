# Thesis Code Repository

This repository contains the implementation of the algorithms and simulations used in my thesis: **"Event-Triggered Methods for Reduced Function Evaluation in Neural Network Controllers"**

## Dependencies

This project uses [`uv`](https://github.com/astral.sh/uv) for dependency management. Follow the steps below to set up the environment.

## Setup Instructions

### 1. Install `uv`
If you don’t have `uv` installed, install it with:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### 2. Install dependencies
Run:
```sh
uv sync
```
### 3. Activate the Virtual Environment
If you need to manually activate it:
```sh
source .venv/bin/activate
```

## Mosek License Requirement
This project requires a valid Mosek license.  

- If you already have a license, ensure it is accessible via the `MOSEKLM_LICENSE_FILE` environment variable or placed in the default Mosek license path.  
    - If you need a license, you can obtain a free academic license from [Mosek’s website](https://www.mosek.com/products/academic-licenses/).

## Repository Structure

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
