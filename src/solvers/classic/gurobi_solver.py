# src/solvers/classic/gurobi_solver.py
import time
from typing import Dict, Any
from src.solvers.interface import SolverInterface
from src.utils.generator import load_instance_from_file

try:
    from gurobipy import Model, GRB
except ImportError:
    # This allows the program to run even if gurobi is not installed,
    # as long as it's not selected in the config.
    GRB = None
    Model = None

class GurobiSolver(SolverInterface):
    """
    A solver for the 0-1 Knapsack Problem using the Gurobi optimizer.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "Gurobi"
        if Model is None:
            raise ImportError("GurobiPy is not installed. Please install it to use this solver.")

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        n = len(weights)
        start_time = time.time()

        model = Model("Knapsack")
        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        model.setObjective(sum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
        model.addConstr(sum(weights[i] * x[i] for i in range(n)) <= capacity, "Capacity")
        model.setParam('OutputFlag', 0)
        model.optimize()
        
        end_time = time.time()
        
        solution = [int(v.X) for v in x.values()]

        return {
            "value": int(model.objVal),
            "time": end_time - start_time,
            "solution": solution
        }