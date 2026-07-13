import os
import numpy as np
import pandas as pd
import random
from typing import Tuple, Dict, Any, Optional
from src.config import RL_MODEL_PATH, RL_CONFIG
from src.rl_agent.env import OptimalExecutionEnv

# Fallback check for Stable Baselines 3
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    PPO = None
    SB3_AVAILABLE = False

class RLExecutionTrainer:
    def __init__(self, model_path: str = RL_MODEL_PATH) -> None:
        self.model_path: str = model_path

    def train_agent(
        self,
        total_timesteps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        total_volume: float = 500.0,
        total_steps: int = 15,
        mid_price: float = 60000.0,
        depth_scale: float = 12.0,
        otc_pct: float = 0.0
    ) -> bool:
        """
        Train PPO RL Agent on OptimalExecutionEnv and save model.
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is not installed or available in this environment. "
                "Please run pip install -r requirements.txt first."
            )
            
        timesteps = total_timesteps if total_timesteps is not None else int(RL_CONFIG["total_timesteps"])
        lr = learning_rate if learning_rate is not None else float(RL_CONFIG["learning_rate"])

        env = OptimalExecutionEnv(
            total_volume=total_volume,
            total_steps=total_steps,
            mid_price=mid_price,
            depth_scale=depth_scale,
            otc_pct=otc_pct
        )

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=int(RL_CONFIG["n_steps"]),
            batch_size=int(RL_CONFIG["batch_size"]),
            gamma=float(RL_CONFIG["gamma"]),
            verbose=1
        )

        model.learn(total_timesteps=timesteps)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save(self.model_path)
        return True

    def run_simulation(
        self,
        total_volume: float = 500.0,
        total_steps: int = 15,
        starting_mid_price: float = 60000.0,
        strategy: str = "rl",
        depth_scale: float = 12.0,
        otc_pct: float = 0.0,
        seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run a single execution episode using the specified strategy.
        Returns:
          - df: pandas DataFrame containing the step-by-step history
          - metrics: dict of summary metrics
        """
        if seed is not None:
            import random
            import numpy as np
            import torch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        env = OptimalExecutionEnv(
            total_volume=total_volume,
            total_steps=total_steps,
            mid_price=starting_mid_price,
            depth_scale=depth_scale,
            otc_pct=otc_pct
        )
        obs, _ = env.reset()
        
        model = None
        if strategy == "rl" and SB3_AVAILABLE:
            model_file = self.model_path + ".zip"
            if os.path.exists(model_file) or os.path.exists(self.model_path):
                try:
                    model = PPO.load(self.model_path)
                except Exception as e:
                    pass

        terminated = False
        truncated = False
        
        arrival_price = env.arrival_price
        total_revenue = 0.0
        total_sold = 0.0

        while not (terminated or truncated):
            if strategy == "rl" and model is not None:
                action, _states = model.predict(obs, deterministic=True)
            elif strategy == "twap":
                steps_left = total_steps - env.current_step
                if steps_left > 0:
                    fraction_of_remaining = 1.0 / steps_left
                else:
                    fraction_of_remaining = 1.0
                action = np.array([fraction_of_remaining], dtype=np.float32)
            else:
                action = np.array([random.uniform(0.05, 0.15)], dtype=np.float32)

            obs, reward, terminated, truncated, _ = env.step(action)

        history = env.history
        df = pd.DataFrame(history)
        if len(df) > 0:
            total_revenue = (df["filled_qty"] * df["vwap"]).sum()
            total_sold = df["filled_qty"].sum()
            avg_exec_price = total_revenue / total_sold if total_sold > 0 else 0.0
            slippage_pct = ((arrival_price - avg_exec_price) / arrival_price) * 100.0
        else:
            avg_exec_price = 0.0
            slippage_pct = 0.0

        metrics = {
            "strategy": strategy,
            "arrival_price": arrival_price,
            "avg_execution_price": avg_exec_price,
            "total_revenue_usd": total_revenue,
            "total_sold_btc": total_sold,
            "total_slippage_pct": slippage_pct,
            "initial_volume_btc": total_volume,
            "horizon_steps": total_steps
        }

        return df, metrics

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    trainer = RLExecutionTrainer()
    trainer.train_agent(total_timesteps=1000)
