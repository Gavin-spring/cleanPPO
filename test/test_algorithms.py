# tests/test_algorithms.py

import pytest

from algorithms import (
    knapsack_01_1d,
    knapsack_01_2d,
    knapsack_branch_and_bound,
    knapsack_gurobi
)

# --- Classic Unit Tests with known answers ---

# Algorithms to be tested for correctness
correctness_test_functions = [
    knapsack_01_1d,
    knapsack_01_2d,
    knapsack_branch_and_bound
]

@pytest.mark.parametrize("algorithm_func", correctness_test_functions)
def test_knapsack_basic_case(algorithm_func):
    """Test a simple, standard knapsack problem instance."""
    weights = [2, 3, 4, 5]
    values = [3, 10, 12, 15]
    capacity = 7
    expected_value = 22 # item with value 10 (weight 3) + item with value 12 (weight 4)
    actual_value = algorithm_func(weights=weights, values=values, capacity=capacity)
    assert actual_value == expected_value

@pytest.mark.parametrize("algorithm_func", correctness_test_functions)
def test_knapsack_all_items_fit(algorithm_func):
    """Test a case where all items can fit in the knapsack."""
    weights = [1, 2, 3]
    values = [10, 20, 30]
    capacity = 10
    expected_value = 60 # 10 + 20 + 30
    actual_value = algorithm_func(weights=weights, values=values, capacity=capacity)
    assert actual_value == expected_value

@pytest.mark.parametrize("algorithm_func", correctness_test_functions)
def test_knapsack_no_items_fit(algorithm_func):
    """
    Test a case where no single item can fit in the knapsack.
    """
    weights = [10, 20, 30]
    values = [100, 200, 300]
    capacity = 5
    expected_value = 0

    actual_value = algorithm_func(weights=weights, values=values, capacity=capacity)

    assert actual_value == expected_value

@pytest.mark.parametrize("algorithm_func", correctness_test_functions)
def test_knapsack_empty_items(algorithm_func):
    """
    Test a case with no items.
    """
    weights = []
    values = []
    capacity = 100
    expected_value = 0

    actual_value = algorithm_func(weights=weights, values=values, capacity=capacity)

    assert actual_value == expected_value


# --- Differential Testing against Gurobi ---

# Define a representative test instance for differential testing
# This should be complex enough to be meaningful but small enough to run quickly
diff_test_instance = {
    "weights": [10, 20, 30, 40, 50, 24, 33, 41],
    "values": [60, 100, 120, 200, 250, 100, 140, 190],
    "capacity": 100
}

# The list of algorithms to compare against Gurobi
algorithms_to_compare = [
    knapsack_01_1d,
    knapsack_01_2d,
    knapsack_branch_and_bound
]

@pytest.mark.parametrize("algorithm_func", algorithms_to_compare)
def test_algorithms_against_gurobi(algorithm_func):
    """
    Compares the result of other algorithms against the Gurobi exact solver.
    This is a differential test.
    """
    # 1. Get the optimal solution from Gurobi
    optimal_value = knapsack_gurobi(
        weights=diff_test_instance["weights"],
        values=diff_test_instance["values"],
        capacity=diff_test_instance["capacity"]
    )
    
    # 2. Run the algorithm being tested
    actual_value = algorithm_func(
        weights=diff_test_instance["weights"],
        values=diff_test_instance["values"],
        capacity=diff_test_instance["capacity"]
    )

    # 3. Compare the results with the Gurobi solution
    assert actual_value == optimal_value, (
        f"Algorithm '{algorithm_func.__name__}' failed differential test. "
        f"Expected: {optimal_value}, Got: {actual_value}"
    )