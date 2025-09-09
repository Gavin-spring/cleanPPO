# src/utils/generator.py
# -*- coding: utf-8 -*-

"""
This module provides core functions to generate and handle different variants
of instances of the knapsack problem. It is configuration-agnostic.
"""

import random
from typing import List, Tuple
import os
import csv
import logging

# Get a logger instance for this module.
# It will inherit the configuration set by the main script.
logger = logging.getLogger(__name__)


def generate_knapsack_instance(
    n: int,
    correlation: str,
    max_weight: int,
    max_value: int,
    capacity_ratio: float
) -> Tuple[List[Tuple[int, int]], int]:
    """
    Generate an instance of the knapsack problem with one constraintion.

    Args:
        n (int): Number of items to generate.
        correlation (str): Type of correlation between item values and weights.
            Options: 'uncorrelated', 'weakly_correlated',
                    'strongly_correlated', 'subset_sum'.
        max_weight (int): Maximum weight for a single item.
        max_value (int): Maximum value for a single item (used when uncorrelated).
        capacity_ratio (float): Ratio of knapsack capacity to the total weight of all items (between 0.0 and 1.0).

    Returns:
        Tuple[List[Tuple[int, int]], int]:
            - A list of items, each represented as a tuple (value, weight).
            - The computed knapsack capacity.
    """
    if correlation not in ['uncorrelated', 'weakly_correlated', 'strongly_correlated', 'subset_sum']:
        raise ValueError("Correlation type must be one of 'uncorrelated', 'weakly_correlated', 'strongly_correlated', or 'subset_sum'")

    items = []
    total_weight = 0

    for _ in range(n):
        weight = random.randint(1, max_weight)
        value = 0

        if correlation == 'uncorrelated':
            value = random.randint(1, max_value)
        elif correlation == 'weakly_correlated':
            noise = int(max_value / 4)
            value = max(1, weight + random.randint(-noise, noise))
        elif correlation == 'strongly_correlated':
            noise = int(max_value / 10)
            value = max(1, weight + random.randint(-noise, noise))
        elif correlation == 'subset_sum':
            value = weight
        
        items.append((value, weight))
        total_weight += weight

    if not (0.0 < capacity_ratio <= 1.0):
        raise ValueError("Capacity ratio must be between 0.0 and 1.0")
    capacity = int(total_weight * capacity_ratio)

    return items, capacity


def save_instance_to_file(items: List[Tuple[int, int]], capacity: int, filepath: str):
    """
    Saves the generated instance to a CSV file.
    Note: This function no longer creates the directory. The calling script is responsible.
    """
    try:
        with open(filepath, 'w', newline='') as f:
            f.write(f"{len(items)} {capacity}\n")
            writer = csv.writer(f)
            writer.writerow(['value', 'weight'])
            for value, weight in items:
                writer.writerow([value, weight])
        logger.debug(f"Instance successfully saved to {filepath}")
    except IOError as e:
        logger.error(f"Failed to save instance to {filepath}: {e}")


def load_instance_from_file(filepath: str) -> Tuple[List[int], List[int], int]:
    """
    Loads a knapsack instance from a CSV file.
    (This function is kept largely the same as your original version)
    """
    weights = []
    values = []
    
    try:
        with open(filepath, 'r', newline='') as f:
            meta_line = f.readline().strip()
            num_items_str, capacity_str = meta_line.split()
            capacity = int(capacity_str)
            expected_num_items = int(num_items_str)

            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                values.append(int(row[0]))
                weights.append(int(row[1]))

        actual_num_items = len(values)
        if actual_num_items != expected_num_items:
            logger.warning(f"Inconsistent data in '{filepath}'. Header specified {expected_num_items}, but file contained {actual_num_items}.")

        logger.debug(f"Instance successfully loaded from {filepath} ({actual_num_items} items).")
        return weights, values, capacity
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return [], [], 0
    except Exception as e:
        logger.error(f"An error occurred while loading {filepath}: {e}")
        return [], [], 0


# TODO: refactor save_instance_to_file and load_instance_from_file to handle multi-dimensional instances
def generate_mckp_instance(
    n: int,
    num_constraints: int,
    max_weights: List[int],
    max_value: int = 1000,
    capacity_ratios: List[float] = None
) -> Tuple[List[Tuple[int, List[int]]], List[int]]:
    """
    generate a multi-dimensional knapsack problem instance.
    Args:
        n (int): Number of items to generate.
        num_constraints (int): Number of constraints (dimensions).
        max_weights (List[int]): Maximum weight for each constraint.
        max_value (int): Maximum value for a single item.
        capacity_ratios (List[float]): Ratios of knapsack capacities to the total weights of all items.

    Returns:
        - items: [(value, [weight1, weight2, ...]), ...]
        - capacities: [capacity1, capacity2, ...]
    """
    if len(max_weights) != num_constraints or len(capacity_ratios) != num_constraints:
        # Ensure that max_weights and capacity_ratios match the number of constraints
        raise ValueError("max_weights and capacity_ratios must have the same length as num_constraints")

    items = []
    total_weights = [0] * num_constraints

    for _ in range(n):
        value = random.randint(1, max_value)
        weights = [random.randint(1, max_w) for max_w in max_weights]
        items.append((value, weights))
        for i in range(num_constraints):
            total_weights[i] += weights[i]

    capacities = [int(total_weights[i] * capacity_ratios[i]) for i in range(num_constraints)]

    return items, capacities

