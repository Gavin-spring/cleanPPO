# src/solvers/ml/feature_extractor.py
import numpy as np
import torch
import os
import logging
from types import SimpleNamespace

from src.utils.generator import load_instance_from_file

logger = logging.getLogger(__name__)

def extract_features_from_instance(instance_path: str, dnn_config: 'SimpleNamespace', generation_config: 'SimpleNamespace') -> torch.Tensor | None:
    """
    Loads a single instance file, normalizes its data, pads it, and returns
    a feature tensor ready for model inference.

    Args:
        instance_path (str): The path to the .csv instance file.

    Returns:
        torch.Tensor | None: A single feature tensor for the model, or None if an error occurs.
    """
    # from src.utils.config_loader import cfg
    try:
        weights, values, capacity = load_instance_from_file(instance_path)
    except Exception as e:
        logger.error(f"Failed to load instance file {instance_path}: {e}")
        return None

    # ML-specific configurations
    gen_cfg = generation_config
    hyperparams = dnn_config.hyperparams
    current_n = len(weights)
    
    # --- START OF NORMALIZATION (Identical to the logic in preprocess_data.py) ---
    weights_norm = np.array(weights, dtype=np.float32) / hyperparams.max_weight_norm
    values_norm = np.array(values, dtype=np.float32) / hyperparams.max_value_norm
    
    value_densities = values_norm / (weights_norm + 1e-6)
    weight_to_capacity_ratios = np.array(weights, dtype=np.float32) / (capacity + 1e-6)
    capacity_ratio_feature = np.full((current_n,), fill_value=gen_cfg.capacity_ratio, dtype=np.float32)
    normalized_capacity = capacity / (hyperparams.max_n * hyperparams.max_weight_norm)

    feature_vector = np.concatenate([
        weights_norm, # Normalized weights, length: max_n
        values_norm, # Normalized values, length: max_n
        value_densities, # Value densities, length: max_n
        weight_to_capacity_ratios,  # Weight to capacity ratios, length: max_n
        capacity_ratio_feature,  # Capacity ratio feature, length: max_n
        [normalized_capacity]   # Normalized capacity, length: 1
    ]).astype(np.float32)

    # --- Padding ---
    padding_size = hyperparams.input_size - len(feature_vector)
    if padding_size < 0:
        logger.warning(f"Feature vector for {instance_path} is too long ({len(feature_vector)} > {hyperparams.input_size}). Truncating.")
        feature_vector = feature_vector[:hyperparams.input_size]
        padding_size = 0
        
    padded_feature_vector = np.pad(feature_vector, (0, padding_size), 'constant')
    
    return torch.tensor(padded_feature_vector)