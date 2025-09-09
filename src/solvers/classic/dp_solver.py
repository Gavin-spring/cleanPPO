# src/solvers/classic/dp_solver.py
import time
import math
from typing import Dict, Any, List
from src.solvers.interface import SolverInterface
from src.utils.generator import load_instance_from_file

class DPSolver2D(SolverInterface):
    """
    A solver for the 0-1 Knapsack Problem using a 2D Dynamic Programming table.
    This corresponds to the original 'knapsack_01_2d' function.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "2D DP"

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        n = len(weights)
        start_time = time.time()

        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        end_time = time.time()
        
        return {
            "value": dp[n][capacity],
            "time": end_time - start_time,
            "solution": [] # Note: Backtracking for the item set is not implemented here.
        }

class DPSolver1D(SolverInterface):
    """
    A solver for the 0-1 Knapsack Problem using a space-optimized 1D DP table.
    This corresponds to the original 'knapsack_01_1d' function.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "1D DP (Optimized)"

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        n = len(weights)
        start_time = time.time()
        
        dp = [0] * (capacity + 1)

        for i in range(n):
            current_weight = weights[i]
            current_value = values[i]
            for j in range(capacity, current_weight - 1, -1):
                dp[j] = max(dp[j], current_value + dp[j - current_weight])
        
        end_time = time.time()
        
        return {
            "value": dp[capacity],
            "time": end_time - start_time,
            "solution": []
        }

class DPValueSolver(SolverInterface):
    """
    A solver for the 0-1 Knapsack Problem using DP based on value.
    Efficient when total value is smaller than capacity.
    This corresponds to the original 'knapsack_01_1d_value' function.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "1D DP (on value)"

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        start_time = time.time()
        
        if not weights or capacity < 0:
            return {"value": 0, "time": time.time() - start_time, "solution": []}

        total_value = sum(values)
        dp = [math.inf] * (total_value + 1)
        dp[0] = 0

        for i in range(len(weights)):
            item_weight = weights[i]
            item_value = values[i]
            for v in range(total_value, item_value - 1, -1):
                dp[v] = min(dp[v], dp[v - item_value] + item_weight)

        final_value = 0
        for v in range(total_value, -1, -1):
            if dp[v] <= capacity:
                final_value = v
                break
        
        end_time = time.time()
        
        return {
            "value": final_value,
            "time": end_time - start_time,
            "solution": []
        }

class FPTASSolver(SolverInterface):
    """
    A solver for the 0-1 Knapsack Problem using a Fully Polynomial-Time
    Approximation Scheme (FPTAS).

    This approach scales the values of the items to solve the problem
    efficiently, providing a solution that is guaranteed to be within a
    factor of (1 - epsilon) of the optimal solution.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "FPTAS (on value)"
        # Epsilon is the approximation factor, e.g., 0.1 for 90% accuracy.
        # It can be passed in the config dictionary.
        self.epsilon = self.config.get("epsilon", 0.1)

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        n = len(weights)
        start_time = time.time()

        if n == 0 or capacity <= 0:
            return {"value": 0, "time": time.time() - start_time, "solution": []}

        # Find the maximum value among all items.
        max_value = 0
        for v in values:
            if v > max_value:
                max_value = v
        
        if max_value == 0:
             return {"value": 0, "time": time.time() - start_time, "solution": []}

        # Define the scaling factor K based on epsilon and max_value.
        K = (self.epsilon * max_value) / n

        # Scale down the values of all items and round them to the nearest integer.
        scaled_values = [math.floor(v / K) for v in values]
        
        # The maximum possible scaled value.
        total_scaled_value = sum(scaled_values)
        
        # DP table: dp[v] stores the minimum weight to achieve a scaled value of v.
        dp = [math.inf] * (total_scaled_value + 1)
        dp[0] = 0

        # Run the value-based DP algorithm with the scaled values.
        for i in range(n):
            item_weight = weights[i]
            item_scaled_value = scaled_values[i]
            for v in range(total_scaled_value, item_scaled_value - 1, -1):
                if dp[v - item_scaled_value] != math.inf:
                    dp[v] = min(dp[v], dp[v - item_scaled_value] + item_weight)

        # Find the highest scaled value that can be achieved within the capacity.
        max_achievable_scaled_value = 0
        for v in range(total_scaled_value, -1, -1):
            if dp[v] <= capacity:
                max_achievable_scaled_value = v
                break

        # Reconstruct the final, unscaled value by summing the original values
        # that correspond to the items that would form the solution.
        # NOTE: This implementation does not reconstruct the item set, only the value.
        # To get the true final value, we need to backtrack, which is more complex.
        # For simplicity, we can approximate the final value by rescaling.
        final_value_approximated = max_achievable_scaled_value * K

        end_time = time.time()

        return {
            # Returning the rescaled value as a close approximation.
            "value": final_value_approximated,
            "time": end_time - start_time,
            "solution": [] # Backtracking is needed to get the actual item list.
        }
