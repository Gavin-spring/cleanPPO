# src/utils/run_utils.py
import datetime
from types import SimpleNamespace

def create_run_name(config: SimpleNamespace) -> str:
    """
    Creates a unique and informative name for an experiment run based on the training mode.

    Args:
        config (SimpleNamespace): The configuration object for the run.

    Returns:
        str: A unique name, e.g., '20250715_105500_RL_n50_lr0.001' or '20250715_105500_DNN_n1000_lr0.001'
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        training_mode = config.ml.training_mode
        
        if training_mode == "DNN":
            # Extract key parameters for DNN
            max_n = config.ml.dnn.generation.end_n
            lr = config.ml.dnn.training.learning_rate
            run_name = f"{timestamp}_DNN_n{max_n}_lr{lr}"
        elif training_mode == "RL":
            # Extract key parameters for RL
            max_n = config.ml.rl.hyperparams.max_n
            lr = config.ml.rl.training.learning_rate
            run_name = f"{timestamp}_RL_n{max_n}_lr{lr}"
        else:
            # Fallback for a known training mode that isn't DNN or RL
            run_name = f"{timestamp}_{training_mode}"
            
    except AttributeError:
        # Fallback for general runs if config structure is missing
        run_name = timestamp
        
    return run_name