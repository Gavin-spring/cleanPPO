# train_model.py
import logging
import os
import torch
# import argparse
from src.utils.config_loader import cfg
from src.utils.logger import setup_logger
# from src.solvers.ml.dnn_solver import DNNSolver
from src.utils.run_utils import create_run_name

# allow PyTorch to use higher precision for matrix multiplications
torch.set_float32_matmul_precision('high')

def main():
    """
    This script initializes and runs the training process for an ML solver,
    saving all artifacts into a unique, timestamped directory.
    """
    # --- 1. Create a unique name and directory for this training run ---
    run_name = create_run_name(cfg)
    run_dir = os.path.join(cfg.paths.artifacts, "runs", "training", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Setup logger to save to the new run-specific directory
    setup_logger(run_name="training_log", log_dir=run_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"--- Starting New Training Run: {run_name} ---")
    logger.info(f"All artifacts for this run will be saved in: {run_dir}")

    # --- 2. Initialize Solver based on config ---
    mode = cfg.ml.training_mode
    solver = None
    
    logger.info(f"Initializing solver for training mode: '{mode}'")

    if mode == "dnn":
        from src.solvers.ml.dnn_solver import DNNSolver
        solver = DNNSolver(config=cfg.ml.dnn, device=cfg.ml.device)

    elif mode == "reinforce":
        from src.solvers.ml.rl_solver import RLSolver
        solver = RLSolver(config=cfg.ml.rl.reinforce, device=cfg.ml.device, compile_model=True)

    elif mode == "actor_critic":
        from src.solvers.ml.rl_solver import RLSolver
        solver = RLSolver(config=cfg.ml.rl.actor_critic, device=cfg.ml.device, compile_model=True)
        
    else:
        raise ValueError(f"Unsupported training mode: '{mode}' found in config.yaml")
    
    # --- 3. Define artifact paths and train ---
    artifact_paths = {
        "model": os.path.join(run_dir, "best_model.pth"),
        "reward_plot": os.path.join(run_dir, "reward_curve.png"),
        "loss_plot": os.path.join(run_dir, "loss_curves.png"),
        "entropy_plot": os.path.join(run_dir, "entropy_curve.png")
    }
    
    solver.train(artifact_paths=artifact_paths)
    
    logger.info(f"--- Training Run {run_name} Finished. ---")

if __name__ == '__main__':
    main()