# src/solvers/ml/dnn_solver.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import logging
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Dict, Any, Tuple, Type, Optional
from types import SimpleNamespace
from .feature_extractor import extract_features_from_instance

from src.solvers.interface import SolverInterface
from .dnn_model import KnapsackDNN
from .data_loader import PreprocessedKnapsackDataset

logger = logging.getLogger(__name__)

class DNNSolver(SolverInterface):
    """
    A full-featured solver for the Knapsack Problem using a Deep Neural Network.
    Refactored for clarity and maintainability.
    """
    def __init__(self, config: SimpleNamespace, device: str, model_path: Optional[str] = None, **kwargs):
        """
        Initializes the DNNSolver.
        Args:
            config: The dnn-specific configuration object.
            device: The device to run the model on ('cuda' or 'cpu').
            model_path (optional): Path to a pre-trained model file for evaluation.
            **kwargs: Catches any other keyword arguments (like 'compile_model') to prevent errors.
        """
        super().__init__(config)
        self.name = "DNN"
        self.device = device
        self.model_path = model_path

        # Log a warning if unexpected arguments are passed, for debugging purposes
        if 'compile_model' in kwargs:
            logger.debug("DNNSolver received 'compile_model' argument but does not use it.")

        self.model = KnapsackDNN(
            input_size=self.config.hyperparams.input_size,
            config=self.config
        ).to(self.device)
        
        # If a model path is provided, load it immediately.
        if self.model_path:
            if os.path.exists(self.model_path):
                logger.info(f"Loading pre-trained model for evaluation from: {self.model_path}")
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            else:
                raise FileNotFoundError(f"Model file specified for DNNSolver not found at: {self.model_path}")
        
        self.model.eval() # Set model to evaluation mode by default
        logger.info(f"{self.name} Solver initialized on device: {self.device}")


    # --- Main Orchestration Method ---
    def train(self, artifact_paths: dict):
        """
        Orchestrates the entire training process by calling helper methods.
        The logic is now high-level and easy to read.
        """
        logger.info(f"--- Starting Training for {self.name} ---")

        # --- Extract paths from the artifact_paths dictionary ---
        model_save_path = artifact_paths.get('model')
        loss_plot_path = artifact_paths.get('loss_plot')

        if not model_save_path or not loss_plot_path:
            logger.error("'model' or 'loss_plot' path not found in artifact_paths dictionary.")
            return
        
        # 1. Setup training components
        criterion, optimizer, scheduler = self._setup_training()

        # 2. Prepare data loaders
        train_loader, val_loader = self._prepare_dataloaders()
        if not train_loader or not val_loader:
            return # Abort if data loading failed

        # 3. Main training loop
        best_val_loss = float('inf')
        history = []
        total_epochs = self.config.training.total_epochs
        
        # Early stopping parameters
        patience = 50 # How many epochs to wait for improvement before stopping
        epochs_no_improve = 0
        
        logger.info(f"Starting training for {total_epochs} epochs...")
        for epoch in range(total_epochs):
            train_loss = self._train_one_epoch(train_loader, criterion, optimizer)
            val_loss = self._validate_one_epoch(val_loader, criterion)
            
            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{total_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_save_path)
                logger.info(f"  -> New best model saved to {model_save_path} (Val Loss: {best_val_loss:.6f})")
                epochs_no_improve = 0  # Reset counter
            else:
                epochs_no_improve += 1  # Increment counter

            # Check if we should stop early
            if epochs_no_improve >= patience:
                logger.info(f"--- Early stopping triggered after {patience} epochs with no improvement. ---")
                break # Exit the training loop

        # 4. Finalize and plot results
        self._plot_loss_curve(pd.DataFrame(history), loss_plot_path)
        logger.info(f"--- Finished Training. Best validation loss: {best_val_loss:.6f} ---")

    # --- Helper Methods ---
    def _setup_training(self) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
        """Initializes and returns the loss function, optimizer, and scheduler."""
        train_cfg = self.config.training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        return criterion, optimizer, scheduler

    def _prepare_dataloaders(self) -> Tuple[DataLoader | None, DataLoader | None]:
        """Loads pre-processed data and prepares PyTorch DataLoaders."""
        from src.utils.config_loader import cfg # Local import for root data path
        
        train_path = os.path.join(cfg.paths.data, "processed_dnn_training.pt")
        val_path = os.path.join(cfg.paths.data, "processed_dnn_validation.pt")
        
        # Check if files exist before trying to load
        if not os.path.exists(train_path):
            logger.error(f"Pre-processed data file not found at: {train_path}")
            logger.error("Please run 'python preprocess_data.py --mode dnn --dataset training' first.")
            return None, None
        if not os.path.exists(val_path):
            logger.error(f"Pre-processed data file not found at: {val_path}")
            logger.error("Please run 'python preprocess_data.py --mode dnn --dataset validation' first.")
            return None, None

        train_dataset = PreprocessedKnapsackDataset(train_path)
        val_dataset = PreprocessedKnapsackDataset(val_path)

        if not train_dataset.data or not val_dataset.data:
            logger.error("Could not load pre-processed datasets or datasets are empty.")
            logger.error("Please run 'preprocess_data.py' successfully first.")
            return None, None
            
        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size)
        return train_loader, val_loader

    def _train_one_epoch(self, data_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """Executes a single training epoch."""
        self.model.train()
        total_loss = 0.0
        for inputs, labels in tqdm(data_loader, desc=f"Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(data_loader)

    def _validate_one_epoch(self, data_loader: DataLoader, criterion: nn.Module) -> float:
        """Executes a single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                total_loss += criterion(outputs, labels).item()
        return total_loss / len(data_loader)

    def solve(self, instance_path: str) -> Dict[str, Any]:
        """
        Uses the loaded neural network to predict the optimal value.
        """
        from src.utils.config_loader import cfg
        if not self.model_path:
            raise RuntimeError("DNNSolver was not initialized with a model_path, cannot solve.")
            
        # The model is already loaded in __init__ and set to eval() mode.
        features_tensor = extract_features_from_instance(
            instance_path, 
            dnn_config=self.config, 
            generation_config=cfg.ml.generation
        )
        
        if features_tensor is None:
            return {"value": -1, "time": 0, "solution": []}
            
        features_tensor = features_tensor.unsqueeze(0).to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            predicted_normalized_value = self.model(features_tensor)
        end_time = time.perf_counter()

        predicted_value = predicted_normalized_value.item() * self.config.hyperparams.target_scale_factor
        
        return {
            "value": predicted_value,
            "time": end_time - start_time,
            "solution": []
        }
        
    def _plot_loss_curve(self, history_df: pd.DataFrame, save_path: str):
        """Helper function to plot and save the loss curve using seaborn."""
        if history_df.empty:
            logger.warning("No training history to plot.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 7))
        
        sns.lineplot(data=history_df, x='epoch', y='train_loss', label='Training Loss')
        sns.lineplot(data=history_df, x='epoch', y='val_loss', label='Validation Loss')
        
        plt.title('Training & Validation Loss Curve', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.yscale('log') # Use a log scale for time, as it can vary greatly
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Loss curve plot saved to {save_path}")