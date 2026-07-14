import os
import numpy as np
import pandas as pd
import random
import requests
from typing import Tuple, Dict, Any, Optional
from src.config import (
    RL_MODEL_PATH, RL_CONFIG, RL_MODEL_PATH_STANDARD,
    RL_MODEL_PATH_STRESS, RL_MODEL_PATH_LIVE, HISTORICAL_DATA_PATH
)
from src.rl_agent.env import OptimalExecutionEnv

# Fallback check for Stable Baselines 3
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    PPO = None
    SB3_AVAILABLE = False

def fetch_historical_btc_prices() -> list:
    """
    Fetch daily close prices for BTC-USD for the last 2 years from Yahoo Finance
    and cache to a local CSV file to optimize performance.
    """
    if os.path.exists(HISTORICAL_DATA_PATH):
        try:
            df = pd.read_csv(HISTORICAL_DATA_PATH)
            if "close" in df.columns:
                prices = df["close"].dropna().tolist()
                if len(prices) > 50:
                    return prices
        except Exception:
            pass

    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD?range=2y&interval=1d"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            raw_prices = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
            prices = [float(p) for p in raw_prices if p is not None]
            if len(prices) > 50:
                os.makedirs(os.path.dirname(HISTORICAL_DATA_PATH), exist_ok=True)
                df = pd.DataFrame({"close": prices})
                df.to_csv(HISTORICAL_DATA_PATH, index=False)
                return prices
    except Exception as e:
        print(f"Error fetching historical BTC prices: {e}")

    # Fallback mock price series if API is down
    return [60000.0 + random.uniform(-1000, 1000) for i in range(500)]

class RLExecutionTrainer:
    def __init__(self, model_path: str = RL_MODEL_PATH) -> None:
        self.model_path: str = model_path
        self.model_paths = {
            "standard": RL_MODEL_PATH_STANDARD,
            "stress": RL_MODEL_PATH_STRESS,
            "live": RL_MODEL_PATH_LIVE
        }

    def train_agent(
        self,
        total_timesteps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        total_volume: float = 500.0,
        total_steps: int = 15,
        mid_price: float = 60000.0,
        depth_scale: float = 12.0,
        otc_pct: float = 0.0,
        agent_type: str = "standard"
    ) -> bool:
        """
        Train PPO RL Agent on OptimalExecutionEnv and save model.
        Supports: standard, stress, live.
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is not installed or available in this environment. "
                "Please run pip install -r requirements.txt first."
            )
            
        timesteps = total_timesteps if total_timesteps is not None else int(RL_CONFIG["total_timesteps"])
        lr = learning_rate if learning_rate is not None else float(RL_CONFIG["learning_rate"])

        # Configure environment mode
        mode = "synthetic"
        historical_prices = None
        if agent_type == "stress":
            mode = "stress"
        elif agent_type == "live":
            mode = "historical"
            historical_prices = fetch_historical_btc_prices()

        env = OptimalExecutionEnv(
            total_volume=total_volume,
            total_steps=total_steps,
            mid_price=mid_price,
            depth_scale=depth_scale,
            otc_pct=otc_pct,
            mode=mode,
            historical_prices=historical_prices
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

        target_model_path = self.model_paths.get(agent_type, self.model_path)
        os.makedirs(os.path.dirname(target_model_path), exist_ok=True)
        model.save(target_model_path)
        return True

    def run_simulation(
        self,
        total_volume: float = 500.0,
        total_steps: int = 15,
        starting_mid_price: float = 60000.0,
        strategy: str = "rl",
        depth_scale: float = 12.0,
        otc_pct: float = 0.0,
        seed: Optional[int] = None,
        agent_type: str = "standard"
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

        # Configure environment mode for simulation
        mode = "synthetic"
        historical_prices = None
        if agent_type == "stress":
            mode = "stress"
        elif agent_type == "live":
            mode = "historical"
            historical_prices = fetch_historical_btc_prices()

        env = OptimalExecutionEnv(
            total_volume=total_volume,
            total_steps=total_steps,
            mid_price=starting_mid_price,
            depth_scale=depth_scale,
            otc_pct=otc_pct,
            mode=mode,
            historical_prices=historical_prices
        )
        obs, _ = env.reset()
        
        model = None
        target_model_path = self.model_paths.get(agent_type, self.model_path)
        if strategy == "rl" and SB3_AVAILABLE:
            model_file = target_model_path + ".zip"
            if os.path.exists(model_file) or os.path.exists(target_model_path):
                try:
                    model = PPO.load(target_model_path)
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
