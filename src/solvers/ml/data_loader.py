# src/solvers/ml/data_loader.py
import torch
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class PreprocessedKnapsackDataset(Dataset):
    """
    A PyTorch Dataset that directly loads a pre-processed list of tensors.
    """
    def __init__(self, preprocessed_data_path: str):
        """
        Args:
            preprocessed_data_path (str): Path to the .pt file containing the dataset.
        """
        try:
            self.data = torch.load(preprocessed_data_path)
            logger.info(f"Successfully loaded {len(self.data)} pre-processed items from {preprocessed_data_path}")
        except FileNotFoundError:
            logger.error(f"Pre-processed data file not found at: {preprocessed_data_path}")
            logger.error("Please run 'preprocess_data.py' first.")
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['features'], item['label']