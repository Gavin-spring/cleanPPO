# knapsack_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import random

from src.utils.generator import load_instance_from_file

class KnapsackEnv(gym.Env):
    def __init__(self, data_dir: str, max_n: int, max_weight: int, max_value: int, use_shaping_reward: bool = False):
        """
        初始化环境。

        Args:
            data_dir (str): 包含 .csv 格式问题实例的目录路径。
            max_n (int): 数据集中所有实例中最大的物品数量。
            max_weight (int): 用于归一化的全局最大物品重量。
            max_value (int): 用于归一化的全局最大物品价值。
        """
        super().__init__()
        
        self.max_n = max_n
        self.max_weight = float(max_weight)
        self.max_value = float(max_value)
        self.use_shaping_reward = use_shaping_reward
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        self.instance_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not self.instance_files:
            raise ValueError(f"在目录 {data_dir} 中没有找到 .csv 实例文件")
        
        self.action_space = spaces.Discrete(self.max_n)

        self.observation_space = spaces.Dict({
            "items": spaces.Box(low=0, high=1, shape=(self.max_n, 2), dtype=np.float32), # 归一化后范围是 [0, 1]
            "capacity": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # 容量用全局max_weight归一化
            "mask": spaces.MultiBinary(self.max_n),
        })

        self.current_instance_path = None
        self.weights = None
        self.values = None
        self.initial_capacity = None
        self.remaining_capacity = None
        self.items_packed = None
        self.n_items = None
        self.next_instance_path = None

    def _reset_from_instance(self, instance_path: str):
        """
        【这是核心逻辑】从指定的实例路径重置环境。
        这个方法包含了所有加载、排序和状态初始化的代码。
        """
        self.current_instance_path = instance_path
        
        weights_list, values_list, capacity_int = load_instance_from_file(self.current_instance_path)
        weights_arr = np.array(weights_list, dtype=np.float32)
        values_arr = np.array(values_list, dtype=np.float32)

        # 按价值重量比 (V/W ratio) 降序排序
        ratio = values_arr / (weights_arr + 1e-9)
        sorted_indices = np.argsort(ratio)[::-1]
        
        self.weights = weights_arr[sorted_indices]
        self.values = values_arr[sorted_indices]

        # 初始化环境状态
        self.n_items = len(self.weights)
        self.initial_capacity = float(capacity_int)
        self.remaining_capacity = self.initial_capacity
        self.items_packed = np.zeros(self.max_n, dtype=bool)

    def manual_set_next_instance(self, instance_path: str):
        """
        【这是评估时用的】设置下一次调用reset()时要加载的特定实例。
        """
        self.next_instance_path = instance_path

    def reset(self, seed=None, options=None):
        """
        【这是唯一的Reset方法】重置环境。
        它会检查是否被手动指定了下一个实例。
        """
        super().reset(seed=seed)

        # 检查是否有预设的实例路径
        if self.next_instance_path:
            instance_path = self.next_instance_path
            self.next_instance_path = None # 用完一次后就清空，恢复随机模式
        else:
            # 如果没有预设，就随机选择一个
            instance_path = random.choice(self.instance_files)
        
        # 调用核心逻辑来加载和初始化
        self._reset_from_instance(instance_path)
        
        # 返回标准的观测和信息
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        is_valid = self._is_action_valid(action)
        
        reward = 0.0
        if is_valid:
            reward = float(self.values[action])
            self.remaining_capacity -= self.weights[action]
            self.items_packed[action] = True

        possible_next_mask = self._get_action_mask()
        terminated = not np.any(possible_next_mask)
        truncated = False
        
        # --- 在回合结束时，加入一个评价“最终方案”好坏的塑形奖励 ---
        if terminated and self.use_shaping_reward:
            final_mask = self.items_packed[:self.n_items]
            final_total_value = np.sum(self.values[final_mask])
            final_total_weight = np.sum(self.weights[final_mask])
            
            # 计算背包填充率 (0到1之间)
            fill_ratio = final_total_weight / self.initial_capacity
            
            # 最终的塑形奖励 = 最终总价值 * (填充率^2) * 奖励权重
            # 填充率的平方是为了加大对“装满”这个行为的激励
            # 0.1 是一个可以调整的超参数，避免这个最终奖励过大，影响了每一步的即时奖励
            final_shaping_reward = final_total_value * (fill_ratio ** 2) * 0.1
            
            # 将塑形奖励加到最后一步的奖励上
            reward += final_shaping_reward

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_action_mask(self):
        mask = np.zeros(self.max_n, dtype=bool)
        # 只在当前实例的真实物品范围内检查
        if self.n_items > 0:
            for i in range(self.n_items):
                if not self.items_packed[i] and self.weights[i] <= self.remaining_capacity:
                    mask[i] = True
        return mask

    def _is_action_valid(self, action: int) -> bool:
        if action >= self.n_items:
            return False
        if self.items_packed[action]:
            return False
        if self.weights[action] > self.remaining_capacity:
            return False
        return True

    def _get_observation(self):
        items_obs = np.zeros((self.max_n, 2), dtype=np.float32)
        
        # 【新策略】使用全局 max_weight 和 max_value 进行归一化
        if self.n_items > 0:
            items_obs[:self.n_items, 0] = self.weights / self.max_weight
            items_obs[:self.n_items, 1] = self.values / self.max_value
        
        # 容量也用 max_weight 归一化，使其与物品重量的尺度保持一致
        capacity_obs = np.array([self.remaining_capacity / self.max_weight], dtype=np.float32)

        mask_obs = self._get_action_mask()
        
        return {"items": items_obs, "capacity": capacity_obs, "mask": mask_obs}

    def _get_info(self):
        active_mask = self.items_packed[:self.n_items]
        return {
            "remaining_capacity": self.remaining_capacity,
            "n_items_packed": np.sum(self.items_packed),
            "total_value": np.sum(self.values[active_mask]),
            "instance_file": self.current_instance_path,
        }

    def render(self):
        """
        visualize the current state of the knapsack environment.
        """
        active_mask = self.items_packed[:self.n_items]

        print("--- Knapsack State ---")
        print(f"Instance: {os.path.basename(self.current_instance_path)}")
        print(f"Capacity: {self.remaining_capacity:.2f} / {self.initial_capacity:.2f}")
        packed_values = self.values[active_mask]
        packed_weights = self.weights[active_mask]
        print(f"Items Packed: {np.sum(self.items_packed)} | Total Value: {np.sum(packed_values):.2f} | Total Weight: {np.sum(packed_weights):.2f}")
        print("----------------------")

    def manual_reset(self, instance_path: str):
        """手动指定下一个要加载的实例路径，用于评估"""
        self.current_instance_path = instance_path

        weights_list, values_list, capacity_int = load_instance_from_file(self.current_instance_path)
        weights_arr = np.array(weights_list, dtype=np.float32)
        values_arr = np.array(values_list, dtype=np.float32)

        ratio = values_arr / (weights_arr + 1e-9)
        sorted_indices = np.argsort(ratio)[::-1]
        
        self.weights = weights_arr[sorted_indices]
        self.values = values_arr[sorted_indices]

        self.n_items = len(self.weights)
        self.initial_capacity = float(capacity_int)
        self.remaining_capacity = self.initial_capacity
        self.items_packed = np.zeros(self.max_n, dtype=bool)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info