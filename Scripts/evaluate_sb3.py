# Scripts/evaluate_sb3.py

import os
import time
import pandas as pd
from tqdm import tqdm
import warnings
import argparse
import datetime
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from src.utils.config_loader import cfg, ALGORITHM_REGISTRY
from src.env.knapsack_env import KnapsackEnv
from src.utils.run_utils import create_run_name
from src.evaluation.plotting import plot_results

# --- Main Function ---
def main():
    # --- 1. model path and stats path ---
    # parser --run-dir
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run directory to evaluate.")
    args = parser.parse_args()

    # Create a unique run name based on the current time and the training run name
    is_deterministic = cfg.ml.testing.is_deterministic
    mode_suffix = "G" if is_deterministic else "S"

    training_run_name = os.path.basename(os.path.normpath(args.run_dir))
    if not training_run_name:
        training_run_name = "unknown_training_run"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_run_name = f"{timestamp}-eval_on-{training_run_name}-{mode_suffix}"
    eval_dir = os.path.join("artifacts_sb3", "evaluation", eval_run_name)

    os.makedirs(eval_dir, exist_ok=True)
    print(f"--- Starting new evaluation run for: {args.run_dir} ---")
    print(f"Evaluation results will be saved in: {eval_dir}")

    model_path = os.path.join(args.run_dir, "models", "best_model.zip")
    stats_path = os.path.join(args.run_dir, "models", "vec_normalize.pkl")

    # Record evaluation configuration
    eval_info = {
        "eval_run_name": eval_run_name,
        "training_run_name": training_run_name,
        "is_deterministic": is_deterministic,
        "data_range": cfg.ml.testing.data_range,
        "batch_size": cfg.ml.testing.batch_size,
        "n_samples": cfg.ml.testing.n_samples,
        "data_path": cfg.paths.data_testing,
        "model_path": model_path,
        "stats_path": stats_path,
    }
    eval_info_path = os.path.join(eval_dir, "run_info.json")
    with open(eval_info_path, 'w') as f:
        json.dump(eval_info, f, indent=4)
    print(f"Evaluation configuration saved to: {eval_info_path}")

    # --- 2. 加载模型 (使用尺寸匹配的临时环境) ---
    print("Step 1: Creating temporary environment with training size to load model...")
    model_train_max_n = cfg.ml.rl.ppo.hyperparams.max_n
    temp_env_kwargs = {
        "data_dir": cfg.paths.data_testing,
        "max_n": model_train_max_n,
        "max_weight": cfg.ml.generation.max_weight,
        "max_value": cfg.ml.generation.max_value,
    }
    temp_env = KnapsackEnv(**temp_env_kwargs)

    print(f"Step 2: Loading model from {model_path}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = PPO.load(model_path, env=temp_env)
    print("Model loaded successfully.")

    # --- 3. 手动创建和校准用于评估的VecNormalize环境 ---
    print("Step 3: Manually setting up full-size normalized environment...")
    
    # a. 创建一个临时的、未包装的VecEnv，仅用于加载统计数据
    temp_vec_env = make_vec_env(lambda: KnapsackEnv(**temp_env_kwargs), n_envs=1)
    loaded_stats = VecNormalize.load(stats_path, temp_vec_env)

    # b. 创建我们真正需要的、足够大的评估环境
    eval_max_n = cfg.ml.rl.ppo.hyperparams.eval_max_n
    eval_env_kwargs = temp_env_kwargs.copy()
    eval_env_kwargs['max_n'] = eval_max_n
    eval_env_unwrapped = make_vec_env(KnapsackEnv, n_envs=1, env_kwargs=eval_env_kwargs)
    
    # c. 用一个全新的VecNormalize来包装大环境
    norm_obs_keys = ["items", "capacity"]
    env = VecNormalize(eval_env_unwrapped, norm_obs=True, norm_reward=False, 
                       gamma=cfg.ml.rl.ppo.training.gamma, norm_obs_keys=norm_obs_keys)

    # d. 【核心修复】进行“数据输血”，逐一处理每个观测的键
    #   ret_rms是标量，可以直接复制
    env.ret_rms = loaded_stats.ret_rms
    #   obs_rms是一个字典，我们需要逐一处理
    for key in norm_obs_keys:
        # 获取训练时保存的统计值的长度
        n_train_obs = loaded_stats.obs_rms[key].mean.shape[0]
        # 将数值复制到新环境对应key的统计对象中
        env.obs_rms[key].mean[:n_train_obs] = loaded_stats.obs_rms[key].mean
        env.obs_rms[key].var[:n_train_obs] = loaded_stats.obs_rms[key].var
        env.obs_rms[key].count = loaded_stats.obs_rms[key].count
    print("Normalization stats manually transferred and padded.")

    # e. 更新模型的“身份证”
    print("Updating model's internal spaces to match new environment...")
    model.observation_space = env.observation_space
    model.action_space = env.action_space
    model.policy.observation_space = env.observation_space
    model.policy.action_space = env.action_space

    # f. 将模型与我们手动校准好的大环境关联
    model.set_env(env)
    print("Environment setup complete.")
    
    # --- 4. Start Evaluation ---
    # 读取评估模式配置
    n_samples = cfg.ml.testing.n_samples if not is_deterministic else 1

    # 根据模式打印提示信息
    if is_deterministic:
        print(f"\n--- Evaluating PPO Agent in DETERMINISTIC mode ---")
    else:
        print(f"\n--- Evaluating PPO Agent in SAMPLING mode ({n_samples} samples/instance) ---")

    ppo_results = []
    test_instances = env.venv.get_attr('instance_files')[0]

    model_capacity = cfg.ml.rl.ppo.hyperparams.eval_max_n
    valid_test_instances = [p for p in test_instances if int(os.path.basename(p).split('_n')[1].split('_')[0]) <= model_capacity]

    for instance_path in tqdm(valid_test_instances, desc="Evaluating PPO Agent"):
        best_value_for_instance = -1.0
        
        start_time = time.time()
        # 循环采样 n_samples 次 (在确定性模式下，n_samples为1)
        for _ in range(n_samples):
            # 每次循环都必须手动设置同一个实例并重置环境
            env.venv.env_method('manual_set_next_instance', instance_path)
            obs = env.reset()
            done = [False]
            
            while not done[0]:
                # 关键：根据配置决定是 deterministic 还是 sampling
                action, _ = model.predict(obs, deterministic=is_deterministic)
                obs, _, done, info = env.step(action)
            
            # 记录本次采样/运行得到的总价值
            current_value = info[0]["total_value"]
            if current_value > best_value_for_instance:
                best_value_for_instance = current_value

        end_time = time.time()
        
        # 使用最后一次运行的 info 来获取文件名等信息
        final_info = info[0] 
        ppo_results.append({
            "instance": final_info["instance_file"],
            "n": int(os.path.basename(final_info["instance_file"]).split('_n')[1].split('_')[0]),
            # 记录多次采样中的最佳值 (在确定性模式下，就是那一次的值)
            "ppo_value": best_value_for_instance,
            "ppo_time": (end_time - start_time), # 这是多次采样的总时间
        })
    ppo_df = pd.DataFrame(ppo_results)

    # --- 4. Evaluate Baseline Solver ---
    print("\n--- Evaluating Baseline Solver ---")    
    BaselineSolverClass = cfg.ml.baseline_algorithm
    baseline_solver_name = None

    for name, solver_class in ALGORITHM_REGISTRY.items():
        if solver_class == BaselineSolverClass:
            baseline_solver_name = name
            break
    
    if BaselineSolverClass and baseline_solver_name:
        print(f"Found baseline solver: {baseline_solver_name}")
        solver_instance = BaselineSolverClass(config={})
        baseline_results = []
        for instance_path in tqdm(test_instances, desc=f"Solving with {baseline_solver_name}"):
            result = solver_instance.solve(instance_path)
            baseline_results.append({
                "instance": instance_path,
                "baseline_value": result.get("value", -1),
                "baseline_time": result.get("time", -1),
            })
        baseline_df = pd.DataFrame(baseline_results)
        
        # --- Combine Results ---
        # clean up instance names for merging
        baseline_df['instance'] = baseline_df['instance'].apply(os.path.basename)
        ppo_df['instance'] = ppo_df['instance'].apply(os.path.basename)
        
        merged_df = pd.merge(ppo_df, baseline_df, on="instance")
        # calculate optimality gap
        merged_df['optimality_gap'] = 1.0 - (merged_df['ppo_value'] / merged_df['baseline_value'])
        
        print("\n--- Combined Evaluation Results ---")
        print(merged_df.head())
        
        # --- Save combined results and plots ---
        final_df = merged_df
        
    else:
        print(f"Baseline solver '{baseline_solver_name}' not found. Skipping error analysis.")
        final_df = ppo_df

    # --- 5. Save the final results ---
    results_path = os.path.join(eval_dir, "evaluation_results_full.csv")
    final_df.to_csv(results_path, index=False)
    
    plots_path = os.path.join(eval_dir, "plots")
    plot_results(final_df, save_dir=plots_path)
    
if __name__ == '__main__':
    main()