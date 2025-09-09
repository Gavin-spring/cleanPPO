# src/solvers/ml/ppo_solver.py

import os
import time
import warnings
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from typing import Dict, Any

from src.solvers.interface import SolverInterface
from src.env.knapsack_env import KnapsackEnv
from src.solvers.ml.custom_policy import KnapsackActorCriticPolicy, KnapsackEncoder

class PPOSolver(SolverInterface):
    """
    A wrapper for a pre-trained Stable Baselines 3 PPO model to make it
    compatible with the project's SolverInterface.
    """
    def __init__(self, model_run_dir: str, is_deterministic: bool = True, **kwargs):
        """
        Initializes the PPO solver by loading the model and environment stats.
        This constructor encapsulates all the complex loading logic from evaluate_sb3.py.
        """
        from src.utils.config_loader import cfg
        super().__init__(config=None) # PPO has its own config logic from the run dir
        self.name = "PPO"
        self.is_deterministic = is_deterministic

        # --- 1. Construct paths ---
        model_path = os.path.join(model_run_dir, "models", "best_model.zip")
        stats_path = os.path.join(model_run_dir, "models", "vec_normalize.pkl")
        if not os.path.exists(model_path) or not os.path.exists(stats_path):
            raise FileNotFoundError(f"Could not find best_model.zip or vec_normalize.pkl in {model_run_dir}")

        # --- 2. Create a temporary environment to load the model ---
        model_train_max_n = cfg.ml.rl.ppo.hyperparams.max_n
        temp_env_kwargs = {
            "data_dir": cfg.paths.data_testing, # Use testing data for the env
            "max_n": model_train_max_n,
            "max_weight": cfg.ml.generation.max_weight,
            "max_value": cfg.ml.generation.max_value,
        }
        
        # Load the model using a compatible policy and temporary env
        policy_kwargs = dict(
            features_extractor_class=KnapsackEncoder,
            features_extractor_kwargs=dict(
                embedding_dim=cfg.ml.rl.ppo.hyperparams.embedding_dim,
                nhead=cfg.ml.rl.ppo.hyperparams.nhead,
                num_layers=cfg.ml.rl.ppo.hyperparams.num_layers
                ),
            critic_type=cfg.ml.rl.ppo.hyperparams.architecture.critic_type,
            n_process_block_iters=cfg.ml.rl.ppo.hyperparams.n_process_block_iters,
        )
        temp_env = make_vec_env(KnapsackEnv, n_envs=1, env_kwargs=temp_env_kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model = PPO(policy=KnapsackActorCriticPolicy, env=temp_env, policy_kwargs=policy_kwargs)

        # --- 3. Manually load model parameters ---
        from stable_baselines3.common.save_util import load_from_zip_file
        _, params, _ = load_from_zip_file(model_path, device=self.model.device)
        if 'policy' not in params:
            raise KeyError(f"Could not find 'policy' key in the saved model parameters. Check the model save file.")
        policy_state_dict = params['policy']

        if any(key.startswith('_orig_mod.') for key in policy_state_dict.keys()):
            cleaned_state_dict = {key.replace('_orig_mod.', ''): value for key, value in policy_state_dict.items()}
            self.model.policy.load_state_dict(cleaned_state_dict)
        else:
            self.model.policy.load_state_dict(policy_state_dict)

        # --- 4. Create and calibrate the final evaluation environment ---
        temp_vec_env_for_stats = make_vec_env(lambda: KnapsackEnv(**temp_env_kwargs), n_envs=1)
        loaded_stats = VecNormalize.load(stats_path, temp_vec_env_for_stats)
        
        eval_env_kwargs = temp_env_kwargs.copy()
        eval_env_kwargs['max_n'] = cfg.ml.rl.ppo.hyperparams.eval_max_n
        eval_env_unwrapped = make_vec_env(KnapsackEnv, n_envs=1, env_kwargs=eval_env_kwargs)
        
        self.env = VecNormalize(
            eval_env_unwrapped, norm_obs=True, norm_reward=False,
            gamma=cfg.ml.rl.ppo.training.gamma, norm_obs_keys=["items", "capacity"]
        )
        
        # "Data transfusion" to calibrate the new, larger environment       
        self.env.ret_rms = loaded_stats.ret_rms
        for key in ["items", "capacity"]:
            n_train_obs = loaded_stats.obs_rms[key].mean.shape[0]
            self.env.obs_rms[key].mean[:n_train_obs] = loaded_stats.obs_rms[key].mean
            self.env.obs_rms[key].var[:n_train_obs] = loaded_stats.obs_rms[key].var
            self.env.obs_rms[key].count = loaded_stats.obs_rms[key].count

        # --- 5. Update model's internal spaces BEFORE setting the new environment ---
        print("Updating model's internal spaces to match the new evaluation environment...")
        self.model.observation_space = self.env.observation_space
        self.model.action_space = self.env.action_space
        self.model.policy.observation_space = self.env.observation_space
        self.model.policy.action_space = self.env.action_space
        
        # Link the fully prepared environment to the loaded model
        self.model.set_env(self.env)


    def solve(self, instance_path: str) -> Dict[str, Any]:
        """
        Solves a single knapsack instance using the loaded PPO model.
        This method implements the abstract method from SolverInterface.
        """
        start_time = time.perf_counter()
        
        # Use the environment's method to load the specific instance
        self.env.env_method('manual_set_next_instance', instance_path)
        obs = self.env.reset()
        
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=self.is_deterministic)
            obs, _, dones, infos = self.env.step(action)
            # In a VecEnv, 'done' is an array. We check the first element.
            done = dones[0]

        # After the loop, the 'infos' dict from the last step contains the final results
        final_info = infos[0]
        final_value = final_info.get("total_value", -1.0)
        
        end_time = time.perf_counter()
        
        # The 'solution' bitmask is not easily retrievable here, so we return empty.
        # The primary goal is comparing value and time.
        return {"value": final_value, "time": end_time - start_time, "solution": []}