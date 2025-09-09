# src/utils/config_loader.py
import yaml
import os
import torch
import logging # <-- Required import
from types import SimpleNamespace
from typing import Dict, Any

# --- 1. Get a logger instance for this specific module ---
# This is the line that was missing.
logger = logging.getLogger(__name__)

# --- Import solver functions here ---
# As you build your solvers in src/solvers/, you will add their imports here.
# For now, we use a try-except block to handle missing dependencies gracefully.
try:
    from src.solvers.classic.gurobi_solver import GurobiSolver
    from src.solvers.classic.dp_solver import DPSolver2D, DPSolver1D, DPValueSolver, FPTASSolver
    from src.solvers.classic.heuristic_solvers import GreedySolver, BranchAndBoundSolver
    from src.solvers.ml.dnn_solver import DNNSolver
    from src.solvers.ml.rl_solver import RLSolver
    from src.solvers.ml.ppo_solver import PPOSolver
    from src.solvers.ml.fast_ppo_solver import FastPPOSolver
except ImportError as e:
    logger.warning(f"Could not import all solvers, some may be unavailable. Error: {e}")
    # Define placeholder classes or functions if needed, or just let the registry be smaller.

# A registry to map algorithm names from YAML to actual Python functions.
# We wrap this in a try-catch as well in case the solver files don't exist yet.
try:
    ALGORITHM_REGISTRY = {
        "Gurobi": GurobiSolver,
        # "2D DP": DPSolver2D,
        # "1D DP (on value)": DPValueSolver,
        # "Greedy": GreedySolver,
        # "DNN": DNNSolver,
        # "PointerNet RL": RLSolver,
        "PPO": PPOSolver,
        # "PPO": FastPPOSolver,  # Batch-oriented PPO solver
        # "Branch and Bound": BranchAndBoundSolver,
        # "1D DP (Optimized)": DPSolver1D,
        # "FPTAS": FPTASSolver,
    }
except NameError:
    # This happens if one of the solver classes could not be imported.
    logger.warning("ALGORITHM_REGISTRY might be incomplete due to import errors.")
    ALGORITHM_REGISTRY = {}


def _post_process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the raw config dict to add dynamic values and absolute paths.
    This function contains all logic that cannot be represented in a static YAML file.
    """
    # --- 1. Define Project Root and Build Absolute Paths ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # a. Build absolute paths
    if 'paths' in config_dict:
        for key, rel_path in config_dict['paths'].items():
            config_dict['paths'][key] = os.path.join(project_root, rel_path)
        config_dict['paths']['root'] = project_root

    # --- 2. Dynamic Calculation of ML Hyperparameters ---
    if 'ml' in config_dict:
        ml_gen_cfg = config_dict['ml']['generation'] 
        dnn_hyper_cfg = config_dict['ml']['dnn']['hyperparams']
        
        # Calculate dynamic DNN hyperparameters using the shared generation config
        max_n = dnn_hyper_cfg['max_n_for_architecture']
        dnn_hyper_cfg['max_n'] = max_n
        dnn_hyper_cfg['input_size'] = max_n * dnn_hyper_cfg['input_size_factor'] + dnn_hyper_cfg['input_size_plus']
        dnn_hyper_cfg['target_scale_factor'] = float(
            max_n * ml_gen_cfg['max_value'] * dnn_hyper_cfg['target_scale_factor_multiplier']
        )

    # --- 3. Auto-detect Hardware Device ---
    if 'ml' in config_dict and config_dict['ml'].get('device') == 'auto':
        config_dict['ml']['device'] = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 4. Map Algorithm Names to Functions ---
    # a. Map the list of algorithms to test
    if 'classic_solvers' in config_dict:
        classic_cfg = config_dict['classic_solvers']
        if 'algorithms_to_test' in classic_cfg:
            available_solvers = [name for name in classic_cfg['algorithms_to_test'] if name in ALGORITHM_REGISTRY]
            classic_cfg['algorithms_to_test'] = {name: ALGORITHM_REGISTRY[name] for name in available_solvers}
    
    # b. Map the baseline algorithm from its NEW location
    if 'ml' in config_dict and 'baseline_algorithm' in config_dict['ml']:
        baseline_name = config_dict['ml'].get('baseline_algorithm')
        if baseline_name and baseline_name in ALGORITHM_REGISTRY:
            config_dict['ml']['baseline_algorithm'] = ALGORITHM_REGISTRY[baseline_name]
        elif baseline_name:
            logger.warning(f"Baseline algorithm '{baseline_name}' defined in config.yaml but not found in ALGORITHM_REGISTRY. Setting to None.")
            config_dict['ml']['baseline_algorithm'] = None

    return config_dict

def load_config(config_path: str = 'configs/config.yaml') -> SimpleNamespace:
    """
    Loads, processes, and returns the project configuration from a YAML file
    as a SimpleNamespace object for dot notation access.
    """
    # Define project root relative to this file's location
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    full_config_path = os.path.join(project_root, config_path)
    
    try:
        with open(full_config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at: {full_config_path}")
    
    # Process the loaded dictionary to add dynamic values
    processed_config = _post_process_config(config_dict)

    # Convert the final dictionary to a SimpleNamespace for easy attribute access
    def dict_to_namespace(d: Dict) -> Any: # Return type can be mixed
        for k, v in d.items():
            if k == 'algorithms_to_test':
                continue
            if isinstance(v, dict):
                d[k] = dict_to_namespace(v)
        return SimpleNamespace(**d)

    return dict_to_namespace(processed_config)

# --- Create a single, global config instance for easy import across the project ---
# Other modules can simply use: from src.utils.config_loader import cfg
cfg = load_config()