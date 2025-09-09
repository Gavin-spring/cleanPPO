# src/solvers/interface.py
from abc import ABC, abstractmethod
from typing import Dict, Any

class SolverInterface(ABC):
    """
    An abstract base class (interface) that all solver classes must implement.
    This ensures that every solver, whether classic or ML-based, can be
    used in a consistent way by the evaluation scripts.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the solver with its configuration.
        
        Args:
            config (Dict[str, Any], optional): A dictionary containing configuration
                                              parameters for the solver. Defaults to None.
        """
        self.config = config if config is not None else {}
        self.name = "Unnamed Solver"

    def train(self, training_data_path: str):
        """
        Train the model, if applicable. For classic algorithms that do not require
        training, this method can be left empty.
        
        Args:
            training_data_path (str): The path to the directory containing training data.
        """
        # Default implementation does nothing, as most classic algorithms don't need training.
        print(f"Solver '{self.name}' does not require training. Skipping.")
        pass

    @abstractmethod
    def solve(self, instance_path: str) -> Dict[str, Any]:
        """
        Solve a given problem instance from a file.
        This method MUST be implemented by all subclasses.

        Args:
            instance_path (str): The path to the file containing the problem instance.

        Returns:
            Dict[str, Any]: A dictionary containing the results, for example:
                            {'solution': [0, 1, 1, ...], 'value': 123, 'time': 0.05}
        """
        pass