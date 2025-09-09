# src/evaluation/reporting.py
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def save_results_to_csv(results_df: pd.DataFrame, save_path: str):
    """
    Saves the aggregated evaluation results DataFrame to a CSV file.

    Args:
        results_df (pd.DataFrame): The DataFrame containing the results.
        save_path (str): The full path to save the CSV file.
    """
    try:
        # Ensure the directory exists before saving
        output_dir = os.path.dirname(save_path)
        os.makedirs(output_dir, exist_ok=True)
        
        results_df.to_csv(save_path, index=False)
        logger.info(f"Evaluation results successfully saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV at {save_path}: {e}")