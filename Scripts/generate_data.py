# generate_data.py
# -*- coding: utf-8 -*-

"""
This is the single entry point for generating all datasets for the project.
It uses the configuration from 'configs/config.yaml' and the core functions
from 'src/utils/generator.py'.
"""

import os
import logging
import random
from tqdm import tqdm # Using tqdm for a nice progress bar

# --- Import project modules ---
from src.utils.config_loader import cfg
from src.utils.logger import setup_logger
import src.utils.generator as gen # Import our new generator library

def create_dataset(
    dataset_name: str,
    output_dir: str,
    instance_params: dict,
    n_range: tuple = None,
    n_fixed: int = None,
    num_instances: int = 1
):
    """
    A generic function to create a dataset of knapsack instances.

    Args:
        dataset_name (str): A name for the generation task (e.g., 'DNN-Training').
        output_dir (str): The directory to save the instance files.
        instance_params (dict): Parameters for the instance generator.
        n_range (tuple): A tuple for varied sizes (start, stop, step).
        n_fixed (int): A fixed size for all instances.
        num_instances (int): The number of instances to generate for each size 'n'.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting dataset generation: '{dataset_name}' ---")
    os.makedirs(output_dir, exist_ok=True)
    
    if n_range:
        # Correct the range to be inclusive of the end value
        range_of_n = range(n_range[0], n_range[1] + 1, n_range[2])
        total_tasks = len(range_of_n) * num_instances
    elif n_fixed:
        range_of_n = [n_fixed]
        total_tasks = num_instances
    else:
        logger.error("Either n_range or n_fixed must be provided.")
        return

    with tqdm(total=total_tasks, desc=f"Generating {dataset_name}") as pbar:
        for n in range_of_n:
            for i in range(num_instances):
                ratio_range = instance_params['capacity_ratio_range']
                randomized_ratio = random.uniform(ratio_range[0], ratio_range[1])
                items, capacity = gen.generate_knapsack_instance(
                    n=n,
                    correlation=instance_params['correlation'],
                    max_weight=instance_params['max_weight'],
                    max_value=instance_params['max_value'],
                    capacity_ratio=randomized_ratio
                )

                filename = os.path.join(output_dir, f"instance_n{n}_{instance_params['correlation']}_{i+1}.csv")
                gen.save_instance_to_file(items, capacity, filename)
                pbar.update(1)

    logger.info(f"--- Dataset generation '{dataset_name}' complete. Files saved in '{output_dir}'. ---")

def main():
    # --- Configure logger ONCE for this script run ---
    setup_logger(run_name="data_generation", log_dir=cfg.paths.logs)

    # --- Control Panel for Generating Datasets ---
    
    # Shared parameters for instance generation, loaded from config
    shared_instance_params = {
        'correlation': cfg.ml.generation.correlation_type,
        'max_weight': cfg.ml.generation.max_weight,
        'max_value': cfg.ml.generation.max_value,
        'capacity_ratio_range': cfg.ml.generation.capacity_ratio_range,
    }
    # Range of 'n' for training and validation sets
    # model_n_range = (cfg.ml.generation.start_n, cfg.ml.generation.end_n, cfg.ml.generation.step_n)

    # ==========================================================
    # # === Tasks for Large Scale Training & Testing ===
    # ==========================================================
    # # === Task 1: Generate the TRAINING set for ML models ===
    # # This set is typically large, with many instances per 'n'.
    # print("Running Task 1: Generate TRAINING set for ML models...")    
    # create_dataset(
    #     dataset_name="ML-Training-Set",
    #     output_dir=cfg.paths.data_training,
    #     instance_params=shared_instance_params,
    #     n_range=model_n_range,
    #     num_instances=500 # e.g., 200 instances per size 'n'
    # )

    # # === Task 2: Generate the VALIDATION set for ML models ===
    # # This set is usually smaller than the training set.
    # print("Running Task 2: Generate VALIDATION set for ML models...")
    # create_dataset(
    #     dataset_name="ML-Validation-Set",
    #     output_dir=cfg.paths.data_validation,
    #     instance_params=shared_instance_params,
    #     n_range=model_n_range,
    #     num_instances=100 # e.g., 50 instances per size 'n'
    # )
    
    # # === Task 3: Generate the common TESTING set for ALL solvers ===
    # # This set is for final benchmarking. Typically has fewer instances per 'n' but may cover a wider range.
    # print("Running Task 3: Generate common TESTING set...")
    # create_dataset(
    #     dataset_name="Final-Testing-Set",
    #     output_dir=cfg.paths.data_testing,
    #     instance_params={ # Using the general-purpose config
    #         'capacity_ratio_range': cfg.data_gen.capacity_ratio_range,
    #         'correlation': cfg.data_gen.correlation_type,
    #         'max_weight': cfg.data_gen.max_weight,
    #         'max_value': cfg.data_gen.max_value,
    #     },
    #     n_range=cfg.data_gen.n_range, # e.g., [10, 601, 10] to test extrapolation
    #     num_instances=150 # e.g., 10 instances per size 'n' for robust testing
    # )
    
    # ==========================================================
    # # === Small-scale Dataset Tasks ===
    # ==========================================================
    # small_n_range = (5, 50, 5)  # Smaller range for quick testing
    # print("Running Task: Generate a small TRAINING set...")
    # create_dataset(
    #     dataset_name="Training-Small-Set",
    #     output_dir="data/small_training",
    #     instance_params=shared_instance_params,
    #     n_range=small_n_range,
    #     num_instances=100,
    # )
    
    # print("Running Task: Generate a small VALIDATION set...")
    # create_dataset(
    #     dataset_name="Validation-Small-Set",
    #     output_dir="data/small_validation",
    #     instance_params=shared_instance_params,
    #     n_range=small_n_range,
    #     num_instances=30,
    # )

    # print("Running Task: Generate a small TESTING set...")
    # create_dataset(
    #     dataset_name="Testing-Small-Set",
    #     output_dir="data/small_testing",
    #     instance_params=shared_instance_params,
    #     n_range=(50, 200, 5),
    #     num_instances=50,
    # )
    
    # ==========================================================
    # ===  Medium-Scale Dataset Tasks ===
    # ==========================================================
    # medium_n_range = (5, 150, 5)
    # print("Running Task: Generate medium-scale datasets for ML models...")
    # # === Training Sets for ML Models ===
    # create_dataset(
    #     dataset_name="ML-Training-Medium",
    #     output_dir=cfg.paths.data_training, # change in config.yaml
    #     instance_params=shared_instance_params,
    #     n_range=medium_n_range,
    #     num_instances=100 # moderate number of instances
    # )

    # # === Validation Sets for ML Models ===
    # create_dataset(
    #     dataset_name="ML-Validation-Medium",
    #     output_dir=cfg.paths.data_validation,
    #     instance_params=shared_instance_params,
    #     n_range=medium_n_range,
    #     num_instances=30 # fewer instances for validation
    # )

    # # === Testing Set for ML Solvers ===
    # create_dataset(
    #     dataset_name="Final-Testing-Set",
    #     output_dir=cfg.paths.data_testing,
    #     instance_params=shared_instance_params,
    #     n_range=(5, 300, 10), # more diverse range for testing
    #     num_instances=50 # not too many instances, but enough for testing
    # )
    
    # ==========================================================
    # ===  Sanity Check ===
    # ==========================================================
    # sanity_n_range = (20, 20, 1)
    # print("Running Task: Generate medium-scale datasets for ML models...")
    # # === Training Sets for ML Models ===
    # create_dataset(
    #     dataset_name="SanityCheck-n20-Training",
    #     output_dir=cfg.paths.data_training, # change in config.yaml
    #     instance_params=shared_instance_params,
    #     n_range=sanity_n_range,
    #     num_instances=100
    # )

    # # === Validation Sets for ML Models ===
    # create_dataset(
    #     dataset_name="SanityCheck-n20-Validation",
    #     output_dir=cfg.paths.data_validation,
    #     instance_params=shared_instance_params,
    #     n_range=sanity_n_range,
    #     num_instances=30
    # )

    # # === Testing Sets for ML Models ===
    # create_dataset(
    #     dataset_name="SanityCheck-n20-Testing",
    #     output_dir=cfg.paths.data_testing,
    #     instance_params=shared_instance_params,
    #     n_range=sanity_n_range,
    #     num_instances=50
    # )

    # ==========================================================
    # ===  Fixed data ===
    # ==========================================================
    fixed_n_range = (100, 1000, 50)
    # print("Running Task: Generate fixed data for ML models...")
    # # === Training Sets for ML Models ===
    # create_dataset(
    #     dataset_name="FIXED-n50-Training",
    #     output_dir=cfg.paths.data_training, # change in config.yaml
    #     instance_params=shared_instance_params,
    #     n_range=fixed_n_range,
    #     num_instances=100
    # )

    # # === Validation Sets for ML Models ===
    # create_dataset(
    #     dataset_name="FIXED-n50-Validation",
    #     output_dir=cfg.paths.data_validation,
    #     instance_params=shared_instance_params,
    #     n_range=fixed_n_range,
    #     num_instances=30
    # )

    # === Testing Sets for ML Models ===
    create_dataset(
        dataset_name="n100-1000-Testing",
        output_dir=cfg.paths.data_testing,
        instance_params=shared_instance_params,
        n_range=fixed_n_range,
        num_instances=10
    )

    print("\nAll selected data generation tasks are complete.")

if __name__ == '__main__':
    main()
