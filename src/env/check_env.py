from stable_baselines3.common.env_checker import check_env
from src.env.knapsack_env import KnapsackEnv

env = KnapsackEnv(data_dir="data/sanity_training", max_n=20, max_weight=100, max_value=100)

check_env(env)
print("Environment is valid!")