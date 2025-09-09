import pickle
import numpy as np
# Although not strictly necessary for loading, importing the class can help with type hinting and clarity
from stable_baselines3.common.vec_env import VecNormalize

# --- 1. Set the path to your file ---
# Make sure to replace this with the actual path to your vec_normalize.pkl
file_path = "artifacts_sb3/training/Exp26-ReplicateExp13-v2-seed42-20250810_014913/models/vec_normalize.pkl"

print(f"Attempting to load file: {file_path}\n")

try:
    # --- 2. Load the file in binary read mode ('rb') ---
    with open(file_path, "rb") as f:
        # The loaded object is an instance of the VecNormalize class
        vec_normalize_stats = pickle.load(f)

    # --- 3. Print all relevant attributes of the loaded object ---
    print("--- Successfully Loaded VecNormalize Object ---")
    print(f"Object type: {type(vec_normalize_stats)}\n")

    print("--- Configuration Parameters ---")
    print(f"Training mode: {vec_normalize_stats.training}")
    print(f"Normalize observations: {vec_normalize_stats.norm_obs}")
    print(f"Normalize rewards: {vec_normalize_stats.norm_reward}")
    print(f"Observation clipping range: ±{vec_normalize_stats.clip_obs}")
    print(f"Reward clipping range: ±{vec_normalize_stats.clip_reward}")
    print(f"Discount factor (gamma): {vec_normalize_stats.gamma}")
    print(f"Epsilon (for division stability): {vec_normalize_stats.epsilon}")
    print(f"Keys to normalize (if DictObs): {vec_normalize_stats.norm_obs_keys}\n")

    print("--- Core Statistics ---")
    # Check if observation normalization was enabled
    if vec_normalize_stats.norm_obs and hasattr(vec_normalize_stats, 'obs_rms'):
        print("📊 Observation Statistics (obs_rms):")
        
        # CORRECTED LOGIC: Check if obs_rms is a dictionary (for DictObs)
        if isinstance(vec_normalize_stats.obs_rms, dict):
            # Iterate through each key in the Dict Observation Space
            for key, rms_object in vec_normalize_stats.obs_rms.items():
                print(f"\n  Statistics for key '{key}':")
                print(f"    - Shape of Mean: {rms_object.mean.shape}")
                print(f"    - Mean (Running Average):\n{rms_object.mean}\n")
                print(f"    - Variance (Running Variance):\n{rms_object.var}\n")
                print(f"    - Sample Count: {rms_object.count}")
        else:
            # Fallback for non-dict observation spaces
            print(f"  - Shape of Mean: {vec_normalize_stats.obs_rms.mean.shape}")
            print(f"  - Mean (Running Average):\n{vec_normalize_stats.obs_rms.mean}\n")
            print(f"  - Variance (Running Variance):\n{vec_normalize_stats.obs_rms.var}\n")
            print(f"  - Sample Count: {vec_normalize_stats.obs_rms.count}\n")
    else:
        print("📊 Observation statistics (obs_rms) not available (normalization was likely disabled).\n")

    # The logic for ret_rms is fine because it's always a single object and rewards are not a dict
    if vec_normalize_stats.norm_reward and hasattr(vec_normalize_stats, 'ret_rms'):
        print("\n📊 Return Statistics (ret_rms):")
        print(f"  - Mean (Running Average): {vec_normalize_stats.ret_rms.mean}")
        print(f"  - Variance (Running Variance): {vec_normalize_stats.ret_rms.var}")
        print(f"  - Sample Count: {vec_normalize_stats.ret_rms.count}\n")
    else:
        # This is what will be printed in your case since norm_reward is False
        print("\n📊 Return statistics (ret_rms) not available (normalization was disabled).\n")
    
    print("--- Internal State Buffers ---")
    # These are used during the step() function
    # old_obs might be large, so we'll just print its shape
    if hasattr(vec_normalize_stats, 'old_obs') and vec_normalize_stats.old_obs is not None:
        print(f"Last observation buffer shape (old_obs): {vec_normalize_stats.old_obs.shape}")
    else:
        print("Last observation buffer (old_obs) is not initialized.")
        
    if hasattr(vec_normalize_stats, 'old_reward') and vec_normalize_stats.old_reward is not None:
        print(f"Last reward buffer shape (old_reward): {vec_normalize_stats.old_reward.shape}")
    else:
        print("Last reward buffer (old_reward) is not initialized.")


except FileNotFoundError:
    print(f"ERROR: File not found at '{file_path}'. Please check the path.")
except ModuleNotFoundError:
    print("ERROR: `stable-baselines3` is not installed in your environment.")
    print("Please run: pip install stable-baselines3")
except Exception as e:
    print(f"An unexpected error occurred: {e}")