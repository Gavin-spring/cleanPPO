# End-to-End Reinforcement Learning for the 0/1 Knapsack Problem

This repository contains the official implementation for the M.Sc. dissertation, *"End-to-End PPO-Based Reinforcement Learning Framework for Scalable 0/1 Knapsack Problem Solving"*.

The project provides an integrated, end-to-end framework for solving the 0/1 Knapsack Problem (KP). It includes modules for procedural data generation, training multiple neural network architectures (including MLP, Pointer Network with REINFORCE, and a state-of-the-art Transformer-PPO model), and performing comparative evaluations against classical algorithms and the commercial Gurobi solver.

## Core Features

* **State-of-the-Art Model**: Implements a powerful deep reinforcement learning agent using a Transformer encoder and Proximal Policy Optimization (PPO) for high stability and performance.
* **Scalable Generalization**: The primary model is designed to be trained on small-scale KP instances (e.g., 5-50 items) and generalize effectively to large, unseen instances (up to 500 items).
* **Integrated Pipeline**: A unified platform that handles the entire research workflow from data generation to final evaluation.
* **Reproducibility**: The entire framework is built with reproducibility in mind, leveraging a centralized configuration system (`config.yaml`) and fixed random seeds.
* **Comparative Analysis**: Includes implementations of baseline models (MLP, REINFORCE) and classical solvers (Dynamic Programming, Greedy) for comprehensive performance comparison.

## System Requirements

* **Operating System**: Linux is highly recommended to leverage the Triton model compiler for accelerated training.
* **Hardware**: An NVIDIA GPU with CUDA version 12.1 or newer is required.
* **Key Dependencies**:
    * `torch`
    * `stable-baselines3`
    * `gymnasium`
    * `numpy` & `pandas`
    * `matplotlib` & `seaborn`
    * `gurobipy`
    * `tqdm`
    * `PyYAML`

A complete list of dependencies can be found in `requirements.txt`.

## Installation

A bash script, `setup_env.sh`, is provided to automate the entire setup process within a Conda environment. This is the recommended installation method.

```bash
bash setup_env.sh
```

This script will:
1.  Install Miniconda.
2.  Create a new Conda environment.
3.  Install all required Python packages.
4.  Install auxiliary tools like Rclone and ngrok for data management and remote monitoring.

## File Structure

```text
.
├── src/                  # Primary source code (solvers, environment, utils)
├── scripts/              # Entry-point scripts for pipeline stages
├── data/                 # Datasets (training, validation, testing)
├── configs/              # Centralized configuration files
├── artifacts/            # Output for MLP, REINFORCE, and classical solvers
├── artifacts_sb3/        # Dedicated output for the PPO model
├── docs/                 # Supplementary documentation
├── requirements.txt      # List of all Python dependencies
├── setup.py              # Project packaging configuration
└── README.md
```

## How to Use the Framework

The project is packaged to provide simple command-line entry points for all major operations.

### Step 1: Data Generation

To generate new problem instances based on the settings in `configs/config.yaml`:
```bash
generate
```

### Step 2: Model Training

The framework supports training three different architectures. All hyperparameters are managed via `configs/config.yaml`.

#### Training the Transformer-PPO Model (Recommended)

This is the primary model presented in the dissertation.
```bash
train-sb3 --name <YourExperimentName>
```

#### Training Baseline Models (MLP or Pointer Network)

To train the MLP or REINFORCE-based Pointer Network, modify the `training_mode` parameter in `configs/config.yaml` to `dnn` or `reinforce`/`actor_critic` respectively.
```bash
train
```

### Step 3: Model Evaluation

#### Dedicated PPO Model Evaluation

To evaluate a trained PPO model against the Gurobi baseline:
```bash
evaluate-sb3 --run-dir /path/to/your/artifacts_sb3/training/ExperimentRun/
```
Results will be saved in `artifacts_sb3/evaluation/`.

#### Unified Solver Evaluation

To compare multiple solvers (classical and neural) in a single run:
```bash
evaluate --dnn-model-path <path> --rl-model-path <path> --ppo-run-dir <path>
```
Results will be saved in `artifacts/results/`.