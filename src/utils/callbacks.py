# src/utils/callbacks.py

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import torch
from tqdm.auto import tqdm


class CompiledModelSaveCallback(EvalCallback):
    """
    A custom callback that handles saving models compiled with torch.compile().
    It temporarily replaces the compiled policy with the original one before saving.
    """
    def _save_model(self) -> None:
        """
        Saves the model, handling the torch.compile wrapper.
        """
        # --- 核心逻辑开始 ---
        # 检查策略是否被编译过 (PyTorch 2.0+ 会有一个 _orig_mod 属性)
        is_compiled = hasattr(self.model.policy, "_orig_mod")
        
        if is_compiled:
            # 1. 保存对已编译模型的引用
            compiled_policy = self.model.policy
            # 2. 临时将模型的策略换成未经编译的原始版本
            self.model.policy = self.model.policy._orig_mod
            print("Swapped to original model for saving.")

        # 调用父类的正常保存逻辑，现在它保存的是“干净”的模型
        super()._save_model()

        if is_compiled:
            # 3. 恢复已编译的模型，以便继续训练
            self.model.policy = compiled_policy
            print("Swapped back to compiled model for continued training.")
        # --- 核心逻辑结束 ---

    def _on_step(self) -> bool:
        # We override _save_model directly, so we just need to ensure the parent's _on_step is called.
        # The parent _on_step will call our custom _save_model when it's time to save.
        continue_training = super()._on_step()
        return continue_training

class TqdmCallback(BaseCallback):
    """
    一个显示tqdm进度条的回调函数。
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.progress_bar = None

    def _on_training_start(self) -> None:
        """在训练开始时被调用"""
        # 从 learn 方法中获取 total_timesteps
        total_timesteps = self.locals.get("total_timesteps", 0)
        self.progress_bar = tqdm(total=total_timesteps, desc="Training")

    def _on_step(self) -> bool:
        """在每一步之后被调用"""
        # 更新进度条
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        return True

    def _on_training_end(self) -> None:
        """在训练结束时被调用"""
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None