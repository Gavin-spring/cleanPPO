# src/solvers/ml/custom_policy.py

import gymnasium as gym
import torch
from torch import nn
from typing import Dict, List, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution

# --- Encoder ---
class KnapsackEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, embedding_dim: int = 128, nhead: int = 4, num_layers: int = 2):
        from src.utils.config_loader import cfg
        super().__init__(observation_space, features_dim=embedding_dim)

        item_feature_dim = cfg.ml.rl.ppo.hyperparams.item_feature_dim  # weight, value
        self.max_possible_n = cfg.ml.rl.ppo.hyperparams.eval_max_n
        self.item_embedder = nn.Linear(item_feature_dim, embedding_dim)

        # [CLS] Token 定义
        self.use_cls_token = cfg.ml.rl.ppo.hyperparams.architecture.use_cls_token
        if self.use_cls_token:
            # 将 [CLS] token 定义为可学习的参数
            self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.max_possible_n + 1, embedding_dim)) # Add [CLS] Token
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        items_obs = observations["items"]
        batch_size, seq_len, _ = items_obs.shape
        item_embeddings = self.item_embedder(items_obs)

        # 根据配置决定是否使用 CLS Token
        if self.use_cls_token:
            # --- 将 [CLS] Token 拼接到每个 batch 序列的最前面 ---
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) # 将 cls_token 复制 batch_size 份
            input_embeddings = torch.cat((cls_tokens, item_embeddings), dim=1)
            
            # 现在序列长度是 seq_len + 1
            input_embeddings += self.positional_encoding[:, :(seq_len + 1), :]
            
            # 将拼接后的序列送入 Transformer
            processed_context = self.transformer_encoder(input_embeddings)
            
            # --- 提取 [CLS] Token 的输出作为全局状态，并分离出物品的 context ---
            pooled_features = processed_context[:, 0]    # 取出第一个 token (CLS) 的输出作为全局状态
            context = processed_context[:, 1:]           # 剩下的部分是物品的 context
            
        else:
            # --- 不使用 [CLS] Token 的原始逻辑 ---
            input_embeddings = item_embeddings + self.positional_encoding[:, :seq_len, :]
            context = self.transformer_encoder(input_embeddings)
            pooled_features = torch.mean(context, dim=1)

        return context, pooled_features

# --- Decoder ---
class PointerDecoder(nn.Module):
    def __init__(self, embedding_dim: int, n_glimpses: int = 1):
        super().__init__()
        self.n_glimpses = n_glimpses
        self.glimpse_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.pointer_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.project_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        # self.project_query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v = nn.Parameter(torch.randn(embedding_dim), requires_grad=True)

    def forward(self, context: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # context: (batch, max_n, embed_dim) - from Encoder
        # query: (batch, embed_dim)

        projected_context = self.project_context(context)
        for _ in range(self.n_glimpses):
            # (batch, 1, embed_dim)
            projected_glimpse_query = self.glimpse_attention(query).unsqueeze(1)
            
            # (batch, max_n)
            glimpse_scores = torch.sum(self.v * torch.tanh(projected_context + projected_glimpse_query), dim=-1)
            glimpse_weights = torch.softmax(glimpse_scores, dim=1)
            
            # 用注意力权重重新聚合context，得到新的query
            # bmm: (batch, 1, max_n) @ (batch, max_n, embed_dim) -> (batch, 1, embed_dim)
            query = torch.bmm(glimpse_weights.unsqueeze(1), context).squeeze(1)

        # 2. Pointer Phase: 使用最终精炼过的query进行决策
        projected_pointer_query = self.pointer_attention(query).unsqueeze(1)
        final_scores = torch.sum(self.v * torch.tanh(projected_context + projected_pointer_query), dim=-1)
        
        return final_scores
        # # (batch, max_n, embed_dim)
        # projected_context = self.project_context(context)
        # # (batch, 1, embed_dim)
        # projected_query = self.project_query(query).unsqueeze(1)
        
        # # (batch, max_n)
        # scores = torch.sum(self.v * torch.tanh(projected_context + projected_query), dim=-1)
        # return scores

# --- Critic ---
# Critic should share the same Encoder as Actor
class SimpleMLPCritic(nn.Module):
    def __init__(self, features_dim: int):
        super().__init__()
        hidden_dim_1 = 256
        hidden_dim_2 = 128
        
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),

            nn.Linear(hidden_dim_2, 1)
        )
    
    def forward(self, pooled_features: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        # 这个Critic只使用pooled_features
        return self.value_net(pooled_features)

class SimpleMLPCritic_d(nn.Module):
    def __init__(self, features_dim: int):
        super().__init__()
        hidden_dim_1 = 512
        hidden_dim_2 = 256
        hidden_dim_3 = 128
        dropout_rate = 0.2
        
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, hidden_dim_1),
            nn.LayerNorm(hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LayerNorm(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.LayerNorm(hidden_dim_3),
            nn.ReLU(),

            nn.Linear(hidden_dim_3, 1)
        )
    
    def forward(self, pooled_features: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        return self.value_net(pooled_features)

class AdvancedAttentionCritic(nn.Module):
    def __init__(self, embedding_dim: int, n_process_block_iters: int):
        from src.utils.config_loader import cfg
        super().__init__()        
        self.n_process_block_iters = n_process_block_iters        
        self.process_block = PointerDecoder(embedding_dim) # 复用PointerDecoder作为注意力层
        self.use_context = cfg.ml.rl.ppo.hyperparams.architecture.critic_input_context

        self.value_decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, pooled_features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # 根据配置决定Critic的输入
        if not self.use_context:
            # 如果不使用context，行为退化为只看全局特征
            return self.value_decoder(pooled_features)

        # pooled_features作为初始的"状态"，进行迭代精炼
        process_block_state = pooled_features
        for _ in range(self.n_process_block_iters):
            # 用当前状态作为query，在所有物品的context上做注意力
            attention_logits = self.process_block(context, process_block_state)
            attention_weights = torch.softmax(attention_logits, dim=1)
            
            # 根据注意力权重，重新聚合context，得到新的、更精炼的状态
            # bmm: (batch, 1, max_n) @ (batch, max_n, embed_dim) -> (batch, 1, embed_dim)
            process_block_state = torch.bmm(attention_weights.unsqueeze(1), context).squeeze(1)
            
        # 3. 使用最终精炼过的状态来预测价值
        return self.value_decoder(process_block_state)

# --- Actor-Critic Policy ---
class KnapsackActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 action_space: gym.spaces.Space,
                 lr_schedule,
                 **kwargs):
        self.critic_type = kwargs.pop("critic_type", "advanced")
        self.n_process_block_iters = kwargs.pop("n_process_block_iters", 3)

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.mlp_extractor = None

    def _build(self, lr_schedule):
        # Actor
        self.action_net = PointerDecoder(self.features_extractor.features_dim)

        # Critic
        if self.critic_type == "simple":
            print(">>> Building with SimpleMLPCritic <<<")
            self.value_net = SimpleMLPCritic(self.features_extractor.features_dim)
        elif self.critic_type == "advanced":
            print(">>> Building with AdvancedAttentionCritic <<<")
            self.value_net = AdvancedAttentionCritic(
                embedding_dim=self.features_extractor.features_dim,
                n_process_block_iters=self.n_process_block_iters
            )
        else:
            raise ValueError(f"Unknown critic_type: {self.critic_type}")
        
        self.action_dist = CategoricalDistribution(self.action_space.n)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_logits_from_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从观测计算出最终的、安全的logits"""
        context, pooled_features = self.extract_features(obs)
        action_logits = self.action_net(context, pooled_features)
        
        mask = obs["mask"].bool()
        action_logits[~mask] = -torch.inf

        # 检查是否存在所有动作都被屏蔽的行
        all_masked_rows = torch.all(~mask, dim=1)
        if all_masked_rows.any():
            # 对于这些完全被屏蔽的行，我们将第一个动作的logit设为0
            # 这可以确保softmax的输出是 [1, 0, 0, ...] 而不是 [nan, nan, nan, ...]
            # 从而避免CUDA崩溃。这个动作是无效的，但可以安全地被采样。
            action_logits[all_masked_rows, 0] = 0

        return action_logits

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shared features encoder
        context, pooled_features = self.extract_features(obs)

        # critic
        if self.critic_type == "simple":
            values = self.value_net(pooled_features)
        else: # advanced
            values = self.value_net(pooled_features, context)

        # actor
        action_logits = self.action_net(context, pooled_features)

        # mask
        mask = obs["mask"].bool()
        action_logits[~mask] = -torch.inf

        all_masked_rows = torch.all(~mask, dim=1)
        if all_masked_rows.any():
            action_logits[all_masked_rows, 0] = 0

        # get results
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        
        return actions, values, log_prob

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context, pooled_features = self.extract_features(obs)
        
        if self.critic_type == "simple":
            values = self.value_net(pooled_features)
        else: # advanced
            values = self.value_net(pooled_features, context)

        action_logits = self.action_net(context, pooled_features)

        mask = obs["mask"].bool()
        action_logits[~mask] = -torch.inf
        all_masked_rows = torch.all(~mask, dim=1)
        if all_masked_rows.any():
            action_logits[all_masked_rows, 0] = 0

        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, log_prob, entropy

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        context, pooled_features = self.extract_features(obs)
        if self.critic_type == "simple":
            return self.value_net(pooled_features)
        else: # advanced
            return self.value_net(pooled_features, context)

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        action_logits = self._get_action_logits_from_obs(observation)
        
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        return distribution.get_actions(deterministic=deterministic)

    def decode_batch(
        self, 
        obs_for_features: Dict[str, torch.Tensor], 
        raw_weights: torch.Tensor, 
        raw_capacity: torch.Tensor, 
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Final, corrected version of autoregressive decoding.
        This version uses the embedding of the previously selected item as the
        query for the next step, which is a standard and robust approach.
        """
        self.eval()
        batch_size = obs_for_features["items"].shape[0]
        max_n = obs_for_features["items"].shape[1]
        
        # --- 1. Run Encoder ONCE ---
        # context contains the rich embeddings of all items.
        context, pooled_features = self.extract_features(obs_for_features)

        # --- 2. Initialize State Tensors ---
        current_mask = obs_for_features["mask"].clone()
        current_capacity = raw_capacity.clone().float()
        actions_sequence = []
        
        # The initial query is the global state representation.
        query = pooled_features

        # --- 3. Internal Decoding Loop with CORRECT Query Update ---
        for _ in range(max_n):
            action_logits = self.action_net(context, query)
            action_logits[~current_mask] = -torch.inf
            
            if deterministic:
                chosen_action = torch.argmax(action_logits, dim=1)
            else:
                distribution = self.action_dist.proba_distribution(action_logits=action_logits)
                chosen_action = distribution.sample()
            
            actions_sequence.append(chosen_action)

            # --- 4. CRITICAL FIX: Update query for the next step ---
            # The query for the next step is the embedding of the item we just chose.
            # This directly informs the model about its last decision.
            # Use torch.gather to select the context vectors for the chosen actions.
            # chosen_action shape: [B], need [B, 1, D] for gather
            chosen_action_expanded = chosen_action.unsqueeze(1).unsqueeze(2).expand(-1, -1, context.shape[2])
            query = torch.gather(context, 1, chosen_action_expanded).squeeze(1)

            # --- 5. Update State Tensors ---
            chosen_weights = torch.gather(raw_weights, 1, chosen_action.unsqueeze(1))
            current_capacity -= chosen_weights.squeeze(1)
            
            capacity_mask = raw_weights <= current_capacity.unsqueeze(1)
            current_mask.scatter_(1, chosen_action.unsqueeze(1), False)
            current_mask = current_mask & capacity_mask
            
            if not current_mask.any():
                break

        return torch.stack(actions_sequence, dim=1)