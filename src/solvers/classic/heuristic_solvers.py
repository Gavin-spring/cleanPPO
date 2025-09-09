# src/solvers/classic/heuristic_solvers.py
import time
from typing import Dict, Any
from queue import PriorityQueue
from src.solvers.interface import SolverInterface
from src.utils.generator import load_instance_from_file

class GreedySolver(SolverInterface):
    """
    An approximation solver for the 0-1 Knapsack Problem using a greedy
    approach based on value-to-weight density. Does not guarantee optimality.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Greedy"

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        n = len(weights)
        start_time = time.time()
        
        items = []
        for i in range(n):
            if weights[i] > 0:
                density = values[i] / weights[i]
                items.append({'id': i, 'value': values[i], 'weight': weights[i], 'density': density})
            elif values[i] > 0:
                 items.append({'id': i, 'value': values[i], 'weight': weights[i], 'density': float('inf')})

        items.sort(key=lambda x: x['density'], reverse=True)

        total_value = 0
        current_weight = 0
        solution = [0] * n
        for item in items:
            if current_weight + item['weight'] <= capacity:
                current_weight += item['weight']
                total_value += item['value']
                solution[item['id']] = 1
        
        end_time = time.time()
        return {"value": total_value, "time": end_time - start_time, "solution": solution}

class BranchAndBoundSolver(SolverInterface):
    """
    An exact solver for the 0-1 Knapsack Problem using Branch and Bound.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Branch and Bound"

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        n = len(weights)
        start_time = time.time()
        
        items = sorted(
            [(values[i], weights[i], values[i] / weights[i] if weights[i] > 0 else float('inf')) for i in range(n)],
            key=lambda x: x[2],
            reverse=True
        )

        def calculate_upper_bound(current_weight, current_value, k):
            bound = current_value
            remaining_capacity = capacity - current_weight
            for i in range(k, n):
                val, w, _ = items[i]
                if w <= remaining_capacity:
                    remaining_capacity -= w
                    bound += val
                else:
                    bound += val * (remaining_capacity / w)
                    break
            return bound

        pq = PriorityQueue()
        initial_upper_bound = calculate_upper_bound(0, 0, 0)
        pq.put((-initial_upper_bound, 0, 0, 0))
        max_value = 0

        while not pq.empty():
            upper_bound_neg, current_value, current_weight, k = pq.get()
            upper_bound = -upper_bound_neg

            if upper_bound < max_value: continue
            if k == n:
                if current_value > max_value: max_value = current_value
                continue

            item_value, item_weight, _ = items[k]

            # Branch 1: Skip item k
            next_upper_bound_skip = calculate_upper_bound(current_weight, current_value, k + 1)
            if next_upper_bound_skip > max_value:
                 pq.put((-next_upper_bound_skip, current_value, current_weight, k + 1))

            # Branch 2: Include item k
            if current_weight + item_weight <= capacity:
                new_weight = current_weight + item_weight
                new_value = current_value + item_value
                if new_value > max_value: max_value = new_value
                
                next_upper_bound_include = calculate_upper_bound(new_weight, new_value, k + 1)
                if next_upper_bound_include > max_value:
                     pq.put((-next_upper_bound_include, new_value, new_weight, k + 1))
        
        end_time = time.time()
        return {"value": max_value, "time": end_time - start_time, "solution": []} # BnB solution tracking is more complex