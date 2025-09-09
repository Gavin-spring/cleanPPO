# # In src/solvers/ml/fast_ppo_solver.py
# # REPLACE the entire file with this FINAL, FULLY CORRECTED version.

# import os
# import time
# import torch
# import torch.nn.functional as F
# import numpy as np
# import pickle
# from typing import Dict, Any, List

# from stable_baselines3 import PPO
# from src.solvers.interface import SolverInterface
# from src.solvers.ml.custom_policy import KnapsackActorCriticPolicy

# class FastPPOSolver(SolverInterface):
#     """
#     A fast, batch-processing PPO solver that uses the policy's custom `decode_batch`
#     method. It manually replicates the full pre-processing pipeline, including
#     correctly handling the dimension mismatch between training and evaluation data,
#     and passes the correct observation format (dictionary) to the policy.
#     """
#     def __init__(self, model_run_dir: str, **kwargs):
#         from src.utils.config_loader import cfg
#         super().__init__(config=None)
#         self.name = "PPO"
#         self.is_deterministic = cfg.ml.testing.is_deterministic
#         device = cfg.ml.device

#         model_path = os.path.join(model_run_dir, "models", "best_model.zip")
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"PPO model not found at {model_path}")
        
#         self.model = PPO.load(
#             model_path, device=device,
#             custom_objects={'policy_class': KnapsackActorCriticPolicy}
#         )
#         self.model.policy.to(device)
#         print(f"FastPPOSolver: Model loaded successfully onto device '{device}'.")

#         stats_path = os.path.join(model_run_dir, "models", "vec_normalize.pkl")
#         self.vec_normalize = None
#         if os.path.exists(stats_path):
#             print(f"Loading VecNormalize stats from: {stats_path}")
#             with open(stats_path, "rb") as f:
#                 self.vec_normalize = pickle.load(f)
#             self.vec_normalize.training = False
#             self.vec_normalize.venv = None
#             if self.vec_normalize.obs_rms:
#                 for key in self.vec_normalize.obs_rms:
#                     rms = self.vec_normalize.obs_rms[key]
#                     rms.mean = torch.from_numpy(rms.mean).to(device).float()
#                     rms.var = torch.from_numpy(rms.var).to(device).float()
#         else:
#             print("Warning: VecNormalize stats not found.")

#     def solve(self, instance_path: str) -> Dict[str, Any]:
#         raise NotImplementedError("FastPPOSolver is designed for batch evaluation.")

#     def solve_batch(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
#         from src.utils.config_loader import cfg
        
#         device = self.model.policy.device
        
#         # --- Step 1: Sorting and Scaling ---
#         raw_weights_unsorted = batch_data['weights'].to(device)
#         raw_values_unsorted = batch_data['values'].to(device)
#         raw_capacity = batch_data['capacity'].to(device)
#         mask_unsorted = batch_data['attention_mask'].to(device)
#         ratio = raw_values_unsorted / (raw_weights_unsorted + 1e-9)
#         ratio[~mask_unsorted] = -1
#         sort_indices = torch.argsort(ratio, dim=1, descending=True)
#         raw_weights = torch.gather(raw_weights_unsorted, 1, sort_indices)
#         raw_values = torch.gather(raw_values_unsorted, 1, sort_indices)
        
#         scaled_features = torch.stack([
#             raw_weights / cfg.data_gen.max_weight,
#             raw_values / cfg.data_gen.max_value
#         ], dim=-1)
#         scaled_capacity = raw_capacity.unsqueeze(1) / cfg.data_gen.max_weight

#         # --- Step 2: Manually Apply Normalization with Dimension Handling ---
#         final_features = scaled_features
#         final_capacity = scaled_capacity
#         if self.vec_normalize and self.vec_normalize.norm_obs and self.vec_normalize.obs_rms:
#             train_max_n = self.vec_normalize.obs_rms['items'].mean.shape[0]

#             features_to_normalize = scaled_features[:, :train_max_n, :]
#             mean, var = self.vec_normalize.obs_rms['items'].mean, self.vec_normalize.obs_rms['items'].var
#             normalized_part = (features_to_normalize - mean) / torch.sqrt(var + self.vec_normalize.epsilon)
            
#             if scaled_features.shape[1] > train_max_n:
#                 remaining_part = scaled_features[:, train_max_n:, :]
#                 final_features = torch.cat([normalized_part, remaining_part], dim=1)
#             else:
#                 final_features = normalized_part

#             mean, var = self.vec_normalize.obs_rms['capacity'].mean, self.vec_normalize.obs_rms['capacity'].var
#             final_capacity = (scaled_capacity - mean) / torch.sqrt(var + self.vec_normalize.epsilon)

#         # --- Step 3: Construct the Observation Dictionary for the Policy ---
#         # This is the crucial fix for the IndexError.
#         obs_dict = {
#             'items': final_features,
#             'capacity': final_capacity,
#             # Mask is not strictly needed by decode_batch but good practice to include
#             'mask': torch.ones(raw_weights.shape, dtype=torch.bool, device=device)
#         }

#         # --- Step 4: Model Inference using the CUSTOM decode_batch ---
#         start_time = time.perf_counter()
#         with torch.no_grad():
#             action_indices_tensor = self.model.policy.decode_batch(
#                 obs_dict, # Pass the dictionary as the first argument
#                 raw_weights, 
#                 raw_capacity, 
#                 deterministic=self.is_deterministic
#             )
#         end_time = time.perf_counter()
#         avg_time_per_instance = (end_time - start_time) / raw_weights.shape[0]

#         # --- Step 5: Post-process ---
#         batch_results = []
#         for i in range(raw_weights.shape[0]):
#             n = batch_data['n'][i].item()
#             solution_indices = action_indices_tensor[i].tolist()
            
#             final_value, final_weight = 0, 0
#             instance_capacity = raw_capacity[i].item()
#             instance_weights_sorted = raw_weights[i]
#             instance_values_sorted = raw_values[i]
#             packed_items = set()
#             for idx in solution_indices:
#                 if idx < n and idx not in packed_items:
#                     if final_weight + instance_weights_sorted[idx] <= instance_capacity:
#                         final_weight += instance_weights_sorted[idx]
#                         final_value += instance_values_sorted[idx]
#                         packed_items.add(idx)
#             batch_results.append({
#                 "value": final_value, "time": avg_time_per_instance,
#                 "solution": [], "total_weight": final_weight
#             })
#         return batch_results


# In src/solvers/ml/fast_ppo_solver.py
# REPLACE the entire file with this FINAL version that includes the crucial clipping step.

import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pickle
from typing import Dict, Any, List

from stable_baselines3 import PPO
from src.solvers.interface import SolverInterface
from src.solvers.ml.custom_policy import KnapsackActorCriticPolicy

class FastPPOSolver(SolverInterface):
    """
    A fast, batch-processing PPO solver that uses the policy's custom `decode_batch`
    method. It manually replicates the full pre-processing pipeline: sorting, scaling,
    normalization, AND clipping, to perfectly match the training conditions.
    """
    def __init__(self, model_run_dir: str, **kwargs):
        from src.utils.config_loader import cfg
        super().__init__(config=None)
        self.name = "PPO"
        self.is_deterministic = cfg.ml.testing.is_deterministic
        device = cfg.ml.device

        model_path = os.path.join(model_run_dir, "models", "best_model.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PPO model not found at {model_path}")
        
        self.model = PPO.load(
            model_path, device=device,
            custom_objects={'policy_class': KnapsackActorCriticPolicy}
        )
        self.model.policy.to(device)
        print(f"FastPPOSolver: Model loaded successfully onto device '{device}'.")

        stats_path = os.path.join(model_run_dir, "models", "vec_normalize.pkl")
        self.vec_normalize = None
        if os.path.exists(stats_path):
            print(f"Loading VecNormalize stats from: {stats_path}")
            with open(stats_path, "rb") as f:
                self.vec_normalize = pickle.load(f)
            self.vec_normalize.training = False
            self.vec_normalize.venv = None
            if self.vec_normalize.obs_rms:
                for key in self.vec_normalize.obs_rms:
                    rms = self.vec_normalize.obs_rms[key]
                    rms.mean = torch.from_numpy(rms.mean).to(device).float()
                    rms.var = torch.from_numpy(rms.var).to(device).float()
        else:
            print("Warning: VecNormalize stats not found.")

    def solve(self, instance_path: str) -> Dict[str, Any]:
        raise NotImplementedError("FastPPOSolver is designed for batch evaluation.")

    def solve_batch(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        from src.utils.config_loader import cfg
        
        device = self.model.policy.device
        eval_max_n = batch_data['weights'].shape[1]

        # --- Step 1: Sorting and Scaling ---
        raw_weights_unsorted = batch_data['weights'].to(device)
        raw_values_unsorted = batch_data['values'].to(device)
        raw_capacity = batch_data['capacity'].to(device)
        mask_unsorted = batch_data['attention_mask'].to(device)
        ratio = raw_values_unsorted / (raw_weights_unsorted + 1e-9)
        ratio[~mask_unsorted] = -1
        sort_indices = torch.argsort(ratio, dim=1, descending=True)
        raw_weights = torch.gather(raw_weights_unsorted, 1, sort_indices)
        raw_values = torch.gather(raw_values_unsorted, 1, sort_indices)
        
        scaled_features = torch.stack([
            raw_weights / cfg.data_gen.max_weight,
            raw_values / cfg.data_gen.max_value
        ], dim=-1)
        scaled_capacity = raw_capacity.unsqueeze(1) / cfg.data_gen.max_weight

        # --- Step 2: Manually Apply Normalization AND CLIPPING ---
        final_features = scaled_features
        final_capacity = scaled_capacity
        if self.vec_normalize and self.vec_normalize.norm_obs and self.vec_normalize.obs_rms:
            train_max_n = self.vec_normalize.obs_rms['items'].mean.shape[0]

            # --- Part A: Process Item Features ---
            features_to_process = scaled_features[:, :train_max_n, :]
            mean, var = self.vec_normalize.obs_rms['items'].mean, self.vec_normalize.obs_rms['items'].var
            # Apply normalization
            processed_part = (features_to_process - mean) / torch.sqrt(var + self.vec_normalize.epsilon)
            # Apply the crucial CLIPPING step
            processed_part = torch.clamp(processed_part, -self.vec_normalize.clip_obs, self.vec_normalize.clip_obs)
            
            if eval_max_n > train_max_n:
                remaining_part = scaled_features[:, train_max_n:, :]
                final_features = torch.cat([processed_part, remaining_part], dim=1)
            else:
                final_features = processed_part

            # --- Part B: Process Capacity ---
            mean, var = self.vec_normalize.obs_rms['capacity'].mean, self.vec_normalize.obs_rms['capacity'].var
            # Apply normalization
            final_capacity = (scaled_capacity - mean) / torch.sqrt(var + self.vec_normalize.epsilon)
            # Apply the crucial CLIPPING step
            final_capacity = torch.clamp(final_capacity, -self.vec_normalize.clip_obs, self.vec_normalize.clip_obs)

        # --- Step 3: Construct Observation Dictionary ---
        obs_dict = {
            'items': final_features,
            'capacity': final_capacity,
            'mask': torch.ones(raw_weights.shape, dtype=torch.bool, device=device)
        }

        # --- Step 4: Model Inference ---
        start_time = time.perf_counter()
        with torch.no_grad():
            action_indices_tensor = self.model.policy.decode_batch(
                obs_dict, raw_weights, raw_capacity, deterministic=self.is_deterministic
            )
        end_time = time.perf_counter()
        avg_time_per_instance = (end_time - start_time) / raw_weights.shape[0]

        # --- Step 5: Post-process ---
        batch_results = []
        for i in range(raw_weights.shape[0]):
            n = batch_data['n'][i].item()
            solution_indices = action_indices_tensor[i].tolist()
            final_value, final_weight = 0, 0
            instance_capacity = raw_capacity[i].item()
            instance_weights_sorted = raw_weights[i]
            instance_values_sorted = raw_values[i]
            packed_items = set()
            for idx in solution_indices:
                if idx < n and idx not in packed_items:
                    if final_weight + instance_weights_sorted[idx] <= instance_capacity:
                        final_weight += instance_weights_sorted[idx]
                        final_value += instance_values_sorted[idx]
                        packed_items.add(idx)
            batch_results.append({
                "value": final_value, "time": avg_time_per_instance,
                "solution": [], "total_weight": final_weight
            })
        return batch_results