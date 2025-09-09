# src/solvers/ml/rl_solver.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple

from src.solvers.interface import SolverInterface
from src.utils.generator import load_instance_from_file
from .data_loader import PreprocessedKnapsackDataset

try:
    from .rl_model import PointerNetwork
except ImportError:
    print("Error: Please ensure the file 'rl_model.py' exists under 'src/solvers/ml/',")
    print("and it contains the 'PointerNetwork' class.")
    exit()

logger = logging.getLogger(__name__)

def knapsack_collate_fn(batch):
    """
    Self defined batch collate function for knapsack problem.
    This function handles padding of weights and values, and creates attention masks.
    """
    # 1. extract weights, values, capacity, and n from the batch
    weights_list = [item['weights'] for item in batch]
    values_list = [item['values'] for item in batch]
    capacity_list = [item['capacity'] for item in batch]
    n_list = [item['n'] for item in batch]
    filenames = [item['filename'] for item in batch]

    # 2. get the maximum length of weights in the batch
    max_len = max(len(w) for w in weights_list)

    padded_weights = []
    padded_values = []
    attention_masks = []

    # 3. pad and create attention masks in batch
    for i in range(len(batch)):
        w = weights_list[i]
        v = values_list[i]
        current_len = len(w)
        
        # Calculate padding length
        padding_len = max_len - current_len

        # use F.pad to pad(right side) the weights and values``
        # format: F.pad(input, (pad_left, pad_right), mode, value)
        padded_w = F.pad(w, (0, padding_len), 'constant', 0)
        padded_v = F.pad(v, (0, padding_len), 'constant', 0)
        padded_weights.append(padded_w)
        padded_values.append(padded_v)

        # create attention mask(1s for actual data, 0s for padding)
        mask = torch.cat([torch.ones(current_len), torch.zeros(padding_len)])
        attention_masks.append(mask)

    # 4. convert lists to tensors
    return {
        'weights': torch.stack(padded_weights),
        'values': torch.stack(padded_values),
        'capacity': torch.stack(capacity_list),
        'n': torch.stack(n_list),
        'attention_mask': torch.stack(attention_masks).bool(), # Convert to boolean tensor
        'filenames': filenames
    }

class RawKnapsackDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, max_weight, max_value):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.max_weight = max_weight
        self.max_value = max_value

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        instance_path = self.files[idx]
        weights, values, capacity = load_instance_from_file(instance_path)

        # --- Sort items by value-to-weight ratio ---
        # Convert weights and values to tensors
        weights_t = torch.tensor(weights, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)

        v_w_ratio = values_t / (weights_t + 1e-9)  # Add a small epsilon to avoid division by zero
        sorted_indices = torch.argsort(v_w_ratio, descending=True)
        sorted_weights = weights_t[sorted_indices]
        sorted_values = values_t[sorted_indices]

        # --- Normalize weights, values and capacity ---
        norm_weights = sorted_weights / self.max_weight
        norm_values = sorted_values / self.max_value
        norm_capacity = torch.tensor(capacity, dtype=torch.float32) / self.max_weight

        # return raw data without padding
        return {
            'weights': norm_weights,
            'values': norm_values,
            'capacity': norm_capacity,
            'n': torch.tensor(len(weights), dtype=torch.int32),
            'filename': os.path.basename(instance_path)
        }

class RLSolver(SolverInterface):
    """
    A reinforcement learning-based solver for the knapsack problem using a pointer network.
    Implements the existing project interface.
    """
    def __init__(self, config, device, model_path=None, compile_model=True):
        super().__init__(config)
        self.name = "PointerNet RL"
        self.device = device

        # 1. Initialize the model (Actor)
        self.model = PointerNetwork(
            config=self.config,
            use_cuda=True if self.device == 'cuda' else False,
            input_dim=2 # Two inputs: weights and values
        ).to(self.device)

        # Compile the model for potential speed-up
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                logger.info("Successfully compiled the model with torch.compile() for potential speed-up.")
            except Exception as e:
                logger.warning(f"Model compilation failed, proceeding without it. Reason: {e}")

        # Load pretrained model if provided
        if model_path:
            if os.path.exists(model_path):
                logger.info(f"Loading pre-trained RL model for evaluation: {model_path}")
                
                # Check if the state_dict contains '_orig_mod.' prefix
                # This is a common prefix when using torch.compile
                state_dict = torch.load(model_path, map_location=self.device)
                if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
                    logger.info("Found compiled model state_dict, stripping '_orig_mod.' prefix...")
                    new_state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}
                    self.model.load_state_dict(new_state_dict, strict=False)
                else:
                    self.model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(f"Model file not found for RLSolver: {model_path}")
        
        self.model.eval() # Set to evaluation mode by default
        logger.info(f"{self.name} solver initialized on device {self.device}")

    # --- Main Orchestration Method ---
    def train(self, artifact_paths: dict):
        """
        Orchestrates the entire training process, now passing the scheduler
        to the epoch training function.
        """
        logger.info(f"--- Starting RL Training for {self.name} ---")
        
        model_save_path = artifact_paths['model']
        reward_plot_path = artifact_paths['reward_plot']
        loss_plot_path = artifact_paths['loss_plot']
        entropy_plot_path = artifact_paths['entropy_plot']
        
        # 1. Setup training components
        optimizer, scheduler = self._setup_training() # receives the scheduler

        # 2. Prepare data loaders
        train_loader, val_loader = self._prepare_dataloaders()
        if not train_loader or not val_loader:
            return

        # 3. Main training loop
        best_val_reward = -float('inf')
        history = []
        total_epochs = self.config.training.total_epochs

        # Early stopping parameters
        patience = 20  # How many epochs to wait for improvement before stopping.
                    # Can be made a config parameter.
        epochs_no_improve = 0

        rl_algorithm_mode = getattr(self.config.training, 'algorithm_mode', 'actor_critic')
        logger.info(f"Using RL training algorithm mode: {rl_algorithm_mode}")
        logger.info(f"Starting training for {total_epochs} epochs...")
        for epoch in range(total_epochs):
            # --- start training ---
            if rl_algorithm_mode == 'reinforce':
                train_reward, final_baseline = self._train_one_epoch_reinforce(train_loader, optimizer, scheduler)
                critic_loss, actor_loss, entropy = 0.0, 0.0, 0.0
            elif rl_algorithm_mode == 'actor_critic':
                train_reward, critic_loss, actor_loss, entropy = self._train_one_epoch_actor_critic(train_loader, optimizer, scheduler)
                final_baseline = torch.tensor(0.0)
            else:
                raise ValueError(f"Unknown algorithm_mode: '{rl_algorithm_mode}' in config.")
            
            val_reward = self._validate_one_epoch(val_loader)
            
            history.append({
                'epoch': epoch + 1, 
                'train_reward': train_reward, 
                'val_reward': val_reward,
                'critic_loss': critic_loss,
                'actor_loss': actor_loss, # for ppo
                'entropy': entropy, # for ppo
                'baseline': final_baseline.item() if rl_algorithm_mode == 'reinforce' else 0.0 # for reinforce
            })

            logger.info(f"Epoch {epoch+1}/{total_epochs}, Train Reward: {train_reward:.4f}, Val Reward: {val_reward:.4f}, " \
            f"Critic Loss: {critic_loss:.4f}, Entropy: {entropy:.4f}")

            if val_reward > best_val_reward:
                best_val_reward = val_reward
                torch.save(self.model.state_dict(), model_save_path)
                logger.info(f"  -> New best model saved to {model_save_path} (Val Reward: {best_val_reward:.4f})")
                epochs_no_improve = 0  # Reset counter on improvement
            else:
                epochs_no_improve += 1  # Increment counter if no improvement

            # Check if we should stop early
            if epochs_no_improve >= patience:
                logger.info(f"--- Early stopping triggered after {patience} epochs with no improvement. ---")
                break  # Exit the training loop

        # 4. Finalize and plot results
        history_df = pd.DataFrame(history)
        self._plot_reward_curve(history_df, reward_plot_path)
        self._plot_loss_curves(history_df, loss_plot_path)
        self._plot_entropy_curve(history_df, entropy_plot_path)

        logger.info(f"--- Finished Training. Best validation reward: {best_val_reward:.4f} ---")

    # --- Helper Methods ---
    def _setup_training(self):
        """
        Initializes and returns the optimizer and the learning rate scheduler
        by reading parameters from the config.
        """
        train_cfg = self.config.training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=train_cfg.learning_rate)
        
        # Create a learning rate scheduler based on the config
        # This scheduler decreases the LR by a factor of 'lr_decay_rate'
        # every 'lr_decay_step' steps.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(
                train_cfg.lr_decay_step,
                train_cfg.lr_decay_step * 1000, # A large upper bound for milestones
                train_cfg.lr_decay_step
            )),
            gamma=train_cfg.lr_decay_rate
        )
        
        logger.info("Optimizer and LR Scheduler have been set up.")
        return optimizer, scheduler

    def _prepare_dataloaders(self):
        """Loads raw data and prepares PyTorch DataLoaders using a collate_fn."""
        from src.utils.config_loader import cfg

        logger.info("Preparing data loaders for RL training...")
        try:
            train_dir = cfg.paths.data_training
            val_dir = cfg.paths.data_validation

            max_weight = cfg.ml.generation.max_weight
            max_value = cfg.ml.generation.max_value

            train_dataset = RawKnapsackDataset(train_dir, max_weight, max_value)
            val_dataset = RawKnapsackDataset(val_dir, max_weight, max_value)
            
            # --- Use the knapsack_collate_fn defined above ---
            train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True, collate_fn=knapsack_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=self.config.training.batch_size, collate_fn=knapsack_collate_fn)
            
            return train_loader, val_loader
        except Exception as e:
            logger.error(f"Failed to prepare dataloaders for RL training. Error: {e}", exc_info=True)
            return None, None

    def _train_one_epoch_reinforce(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler) -> Tuple[float, torch.Tensor]:

        """
        Executes a single training epoch for the RL model,
        now including gradient clipping and scheduler stepping.
        """
        self.model.train() # Set model to training mode
        self.model.decoder.decode_type = 'stochastic'

        total_epoch_reward = 0.0
        # Use baseline_beta from the config
        beta = self.config.training.baseline_beta
        # Initialize baseline on the correct device
        baseline = torch.zeros(1, device=self.device)

        # 1. Use a GradScaler for mixed precision training
        # AMP (Automatic Mixed Precision) taking adventages of Tensor Cores is a feature in PyTorch that allows you to use lower precision (float16) for training,
        # For NVIDIA GPU（RTX 20XX），Tensor Cores dealing with float16 as multiple times faster than float32, and can train larger models.
        # For Pointer Networks and Transformer models, Computationally intensive operations (such as matrix multiplication nn.Linear, torch.bmm) 
        # are typically stable and efficient under float16, and they account for more than 95% of the model's computation time. 
        # Numerically sensitive operations  (such as softmax, log in loss functions, or activation functions with large input ranges) are only a minority . 

        scaler = torch.amp.GradScaler()
        for i, batch_data in enumerate(tqdm(data_loader, desc="Training (REINFORCE)", leave=False)):
            
            weights = batch_data['weights'].to(self.device)
            values = batch_data['values'].to(self.device)
            capacity = batch_data['capacity'].to(self.device)
            attention_mask = batch_data['attention_mask'].to(self.device)
            
            inputs = torch.stack([weights, values], dim=1)            
            inputs_for_model = inputs.permute(0, 2, 1) # Shape: (batch_size, feature_num, max_n)
            
            optimizer.zero_grad()
            
            # 2. Perform forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda'):
                probs_list, action_idxs, _ = self.model(inputs_for_model, capacity, attention_mask)
                rewards = self._calculate_reward(action_idxs, batch_data).to(self.device)

                # 3. Calculate the baseline using an exponential moving average
                if i == 0 and baseline.sum() == 0:
                    baseline = rewards.mean()
                else:
                    baseline = baseline * beta + (1. - beta) * rewards.mean()
                    
                # 4. Calculate the advantage
                advantage = rewards - baseline.detach()

                # Calculate the loss
                log_probs_of_actions = 0
                for prob_dist, action_idx in zip(probs_list, action_idxs):
                    log_prob = torch.log(prob_dist.gather(1, action_idx.unsqueeze(1)).squeeze(1))
                    log_prob[log_prob < -1000] = 0.0
                    log_probs_of_actions += log_prob

                loss = -(log_probs_of_actions * advantage).mean()

            # 5. Scale the loss for mixed precision
            scaler.scale(loss).backward()

            # optimizer.zero_grad() # used when not using AMP
            # loss.backward() # used when not using AMP

            # --- APPLY GRADIENT CLIPPING ---
            # Use the max_grad_norm value from the config
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )

            # optimizer.step() # used when not using AMP

            # --- STEP THE LEARNING RATE SCHEDULER ---
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            total_epoch_reward += rewards.mean().item()
        
        return total_epoch_reward / len(data_loader), baseline

    def _train_one_epoch_actor_critic(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler) -> Tuple[float, float, float, float]:
        self.model.train()

        # --- 1. 从self.config加载超参数 ---
        ppo_cfg = self.config.training
        ppo_epochs = ppo_cfg.ppo_epochs
        gamma = ppo_cfg.gamma
        gae_lambda = ppo_cfg.gae_lambda
        clip_param = ppo_cfg.clip_param
        value_loss_coef = ppo_cfg.value_loss_coef
        entropy_coef = ppo_cfg.entropy_coef

        # 在每个epoch开始时，我们只处理一个batch的数据作为一次完整的PPO迭代
        for batch_data in tqdm(data_loader, desc="PPO Iteration", leave=False):
            
            # --- 2. 经验采集 (Rollout) ---
            # 使用当前的策略(θ_old)与环境交互，采集一批经验数据
            with torch.no_grad():
                # 将数据移动到设备
                weights = batch_data['weights'].to(self.device)
                values = batch_data['values'].to(self.device)
                capacity = batch_data['capacity'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                inputs = torch.stack([weights, values], dim=1).permute(0, 2, 1)

                # 模型前向传播，获得动作、概率、和状态价值
                self.model.decoder.decode_type = 'stochastic'
                # 注意：您的模型现在返回三个值
                probs_list, action_idxs, state_values = self.model(inputs, capacity, attention_mask)

                # 计算旧策略下，采取这些动作的对数概率
                log_probs_of_actions_old = 0
                for prob_dist, action_idx in zip(probs_list, action_idxs):
                    prob_of_action = prob_dist.gather(1, action_idx.unsqueeze(1)).squeeze(1)
                    log_probs_of_actions_old += torch.log(prob_of_action + 1e-9) # 防止log(0)错误
                
                # 计算整个回合的奖励 (Gt)
                rewards = self._calculate_reward(action_idxs, batch_data).to(self.device)

            # --- 3. 计算优势函数 (通用GAE实现) ---
            # 您的场景: T步决策，只有第T步才有奖励R，其余奖励为0
            num_steps = len(action_idxs)
            advantages = torch.zeros_like(rewards)
            last_gae_lam = 0
            
            # 我们只有一个最终奖励，所以V(s_T) = R, V(s_{T+1}) = 0
            # 最后一个时间步的TD误差: delta_T = R_T + gamma * 0 - V(s_T) = R - V(s_T)
            # 注意: state_values 是 V(s_0)，不是 V(s_T)。
            # 对于这种“延迟奖励”问题，一个简单有效的处理方式仍然是:
            advantages = rewards - state_values.detach()
            # 回报目标 (Returns-to-go)
            returns = advantages + state_values.detach()
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            assert not torch.isnan(returns).any() and not torch.isnan(advantages).any(), "NaN detected in returns or advantages"
            # 归一化优势函数
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # --- 4. 优化阶段 (K个Epoch的更新) ---
            # 使用采集到的这批数据，重复训练网络K次
            for _ in range(ppo_epochs):
                # 模型再次前向传播，以获取当前策略下的输出
                self.model.decoder.decode_type = 'stochastic'
                new_probs_list, _, new_state_values = self.model(inputs, capacity, attention_mask)
                
                # 计算新策略下的对数概率和熵
                log_probs_of_actions_new = 0
                entropy = 0
                for prob_dist, action_idx in zip(new_probs_list, action_idxs):
                    prob_of_action = prob_dist.gather(1, action_idx.unsqueeze(1)).squeeze(1)
                    log_probs_of_actions_new += torch.log(prob_of_action + 1e-9)
                    entropy -= (prob_dist * torch.log(prob_dist.clamp(min=1e-9))).sum(dim=1)
                
                # --- 计算PPO损失 ---
                # 1. 概率比
                ratio = torch.exp(log_probs_of_actions_new - log_probs_of_actions_old)
                assert not torch.isnan(ratio).any() and not torch.isinf(ratio).any(), "NaN or Inf detected in ratio"
                
                # 2. Actor损失 (L_CLIP)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 3. Critic损失 (L_VF)
                critic_loss = F.mse_loss(new_state_values, returns)
                
                # 4. 总损失
                loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy.mean()
                assert not torch.isnan(loss).any(), "NaN detected in final loss"

                # --- 梯度更新 ---
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                optimizer.step()
        
        # 返回一些日志信息
        return rewards.mean().item(), critic_loss.item(), actor_loss.item(), entropy.mean().item()

    def _validate_one_epoch(self, data_loader: DataLoader) -> float:
        """Executes a single validation epoch."""
        self.model.eval() # Set model to evaluation mode
        self.model.decoder.decode_type = 'greedy' # Use greedy decoding for validation

        total_val_reward = 0.0
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Validating", leave=False): # I've added tqdm here for a progress bar
                weights = batch_data['weights'].to(self.device)
                values = batch_data['values'].to(self.device)
                capacity = batch_data['capacity'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                
                inputs = torch.stack([weights, values], dim=1)
                # This permute should be consistent with the one in training
                inputs_for_model = inputs.permute(0, 2, 1) 
                
                _ , action_idxs, _ = self.model(inputs_for_model, capacity, attention_mask)                

                rewards = self._calculate_reward(action_idxs, batch_data)
                total_val_reward += rewards.mean().item()
                
        return total_val_reward / len(data_loader)
        
    def _plot_reward_curve(self, history_df: pd.DataFrame, save_path: str):
        """Helper function to plot and save the reward curve."""
        if history_df.empty:
            logger.warning("No training history to plot.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 7))
        
        sns.lineplot(data=history_df, x='epoch', y='train_reward', label='Training Average Reward')
        sns.lineplot(data=history_df, x='epoch', y='val_reward', label='Validation Average Reward')
        # Add a line for the baseline value
        if 'baseline' in history_df.columns:
            sns.lineplot(data=history_df, x='epoch', y='baseline', label='Reward Baseline (EMA)', linestyle='--', color='gray')
        
        plt.title('Training & Validation Reward Curve', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Reward (Total Value)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Reward curve plot saved to {save_path}")

    def _plot_loss_curves(self, history_df: pd.DataFrame, save_path: str):
        """Helper function to plot and save the actor and critic loss curves."""
        if history_df.empty or 'critic_loss' not in history_df.columns:
            logger.warning("No loss history to plot.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 7))
        
        # Plot Critic Loss
        sns.lineplot(data=history_df, x='epoch', y='critic_loss', label='Critic Loss (Value Function)')
        
        # Check if actor_loss is available and plot it
        if 'actor_loss' in history_df.columns:
            # We can use a secondary y-axis if the scales are very different, but for now we'll plot on the same one.
            sns.lineplot(data=history_df, x='epoch', y='actor_loss', label='Actor Loss (Policy)')
        
        plt.title('Training Loss Curves', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Loss curves plot saved to {save_path}")

    def _plot_entropy_curve(self, history_df: pd.DataFrame, save_path: str):
        """Helper function to plot and save the policy entropy curve."""
        if history_df.empty or 'entropy' not in history_df.columns:
            logger.warning("No entropy history to plot.")
            return

        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(15, 7))
        
        sns.lineplot(data=history_df, x='epoch', y='entropy', label='Policy Entropy')
        
        plt.title('Policy Entropy During Training', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Entropy', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Entropy curve plot saved to {save_path}")

    def _calculate_reward_unpack(self, selected_indices, instance_batch):
        """
        Given the item selection sequence output by the model, compute the total value as the reward.
        This function handles batched input.
        """
        batch_size = selected_indices[0].size(0)
        batch_rewards = []

        for i in range(batch_size):
            capacity = instance_batch['capacity'][i].item()
            weights = instance_batch['weights'][i]
            values = instance_batch['values'][i]

            current_weight = 0.0
            current_value = 0.0
            
            # Convert output indices to Python list
            item_priority_list = [idx[i].item() for idx in selected_indices]
            
            packed_items = set() # Prevent duplicate selections
            
            for item_idx in item_priority_list:
                if item_idx in packed_items:
                    continue # Skip already added items
                
                if current_weight + weights[item_idx] <= capacity:
                    current_weight += weights[item_idx]
                    current_value += values[item_idx]
                    packed_items.add(item_idx)
            
            batch_rewards.append(current_value)

        return torch.tensor(batch_rewards, dtype=torch.float32)

    def _calculate_reward(self, selected_indices, instance_batch):
        """
        Given the item selection sequence output by the model, compute the total value as the reward.
        """
        batch_size = selected_indices[0].size(0)
        batch_total_value = torch.zeros(batch_size, device=self.device)
        
        # extract values from the whole instance batch
        # Attention：instance_batch['values'] is padded, so it has shape [batch_size, max_n]
        values_padded = instance_batch['values'].to(self.device)

        # Iterate over each action step in the batch
        for action_step in selected_indices:
            # action_step shape: [batch_size]
            # gather from values_padded to get the values corresponding to the selected indices
            # action_step.unsqueeze(1) -> [batch_size, 1]
            chosen_values = values_padded.gather(1, action_step.unsqueeze(1)).squeeze(1)

            batch_total_value += chosen_values

        return batch_total_value

    def solve(self, instance_path: str):
        """
        Solve a single knapsack problem instance using the trained RL model.
        """
        self.model.eval() # Ensure model is in evaluation mode
        
        # 1. Load and prepare data
        weights, values, capacity = load_instance_from_file(instance_path)
        n_items = len(weights)

        # Create tensors with a batch dimension of 1
        weights_t = torch.tensor(weights, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        values_t = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        capacity_t = torch.tensor([capacity], dtype=torch.float32).to(self.device)

        # Prepare input tensor consistent with training (shape: [1, n, 2])
        input_tensor_original_shape = torch.stack([weights_t, values_t], dim=1).to(self.device)
        input_tensor = input_tensor_original_shape.permute(0, 2, 1) # Shape: (1, feature_num, max_n)

        # 2. create attention_mask
        # For a single instance, there is no padding, so the mask is all True.
        # Shape should be [batch_size, seq_len], which is [1, n_items] here.
        attention_mask = torch.ones(1, n_items, dtype=torch.bool, device=self.device)

        # 3. Model inference (using greedy decoding)
        start_time = time.perf_counter()
        with torch.no_grad():
            # Assume model returns both log probabilities and action indices
            _, action_idxs, _ = self.model(input_tensor, capacity_t, attention_mask) 
        end_time = time.perf_counter()
        
        # 4. Compute final solution
        # `_calculate_reward` can be reused, it returns a tensor of rewards
        instance_data = {'weights': weights_t, 'values': values_t, 'capacity': torch.tensor([capacity])}
        final_value = self._calculate_reward(action_idxs, instance_data).item()
        
        # Extract the list of item indices
        item_indices = [idx[0].item() for idx in action_idxs]
        solution_mask = [0] * len(weights)
        
        # Determine final packing based on selection order
        final_weight = 0
        final_packed_indices = set()
        for idx in item_indices:
            if idx in final_packed_indices:
                continue
            if final_weight + weights[idx] <= capacity:
                final_weight += weights[idx]
                final_packed_indices.add(idx)
        
        for idx in final_packed_indices:
            solution_mask[idx] = 1

        return {
            "value": final_value,
            "time": end_time - start_time,
            "solution": solution_mask
        }

    def solve_batch(self, batch_data):
        """
        Solves a batch of knapsack instances at once.
        """
        self.model.eval()
        self.model.decoder.decode_type = 'greedy'

        # 1. move batch data to device
        weights = batch_data['weights'].to(self.device)
        values = batch_data['values'].to(self.device)
        capacity = batch_data['capacity'].to(self.device)
        attention_mask = batch_data['attention_mask'].to(self.device)
        
        batch_size = weights.size(0)

        # prepare inputs for the model
        inputs_stacked = torch.stack([weights, values], dim=1)
        inputs_for_model = inputs_stacked.permute(0, 2, 1)

        # 2. perform model inference for the entire batch
        start_time = time.perf_counter()
        with torch.no_grad():
            _, action_idxs, _ = self.model(inputs_for_model, capacity, attention_mask)
        end_time = time.perf_counter()

        # Calculate average time per instance
        avg_time_per_instance = (end_time - start_time) / batch_size

        # 3. Calculate rewards for the batch
        rewards = self._calculate_reward(action_idxs, batch_data)

        # 4. Prepare the final results for each instance in the batch
        batch_results = []
        for i in range(batch_size):
            # get raw instance data for the i-th instance
            original_n = batch_data['n'][i].item()
            instance_weights = batch_data['weights'][i][:original_n].tolist()
            
            # extract solution indices from action_idxs
            item_indices = [idx[i].item() for idx in action_idxs]
            solution_mask = [0] * original_n
            
            final_weight = 0
            final_packed_indices = set()
            for idx in item_indices:
                # check if idx is within the original n
                if idx >= original_n:
                    continue
                if idx in final_packed_indices:
                    continue
                if final_weight + instance_weights[idx] <= batch_data['capacity'][i].item():
                    final_weight += instance_weights[idx]
                    final_packed_indices.add(idx)
            
            for idx in final_packed_indices:
                solution_mask[idx] = 1

            batch_results.append({
                "value": rewards[i].item(),
                "time": avg_time_per_instance,
                "solution": solution_mask
            })
        
        return batch_results