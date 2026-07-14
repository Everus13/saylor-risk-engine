import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from src.order_book_sim import OrderBookSimulator

class OptimalExecutionEnv(gym.Env):
    """
    A custom Gymnasium Environment for training RL agents to perform optimal BTC execution.
    The agent needs to sell a specified quantity of BTC over a fixed time horizon
    while minimizing price impact (slippage) and market risk (price variance).
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        total_volume: float = 500.0,  # BTC to sell
        total_steps: int = 15,        # time steps to execute the order
        mid_price: float = 60000.0,   # starting mid price
        depth_scale: float = 12.0,    # L2 book depth scale (liquidity)
        otc_pct: float = 0.0,         # Percentage of sale volume routed via OTC
        mode: str = "synthetic",      # "synthetic", "stress", or "historical"
        historical_prices: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        self.total_volume: float = total_volume
        self.total_steps: int = total_steps
        self.starting_mid_price: float = mid_price
        self.depth_scale: float = depth_scale
        self.otc_pct: float = otc_pct
        self.mode: str = mode
        self.historical_prices: Optional[List[float]] = historical_prices

        # Action Space: continuous value in [0.0, 1.0] representing
        # the fraction of REMAINING volume to sell in this step.
        self.action_space: spaces.Box = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space:
        # 1. remaining_volume_pct [0, 1]
        # 2. remaining_time_pct [0, 1]
        # 3. spread_pct [-1, 1] (rescaled)
        # 4. book_imbalance [-1, 1]
        # 5. price_deviation_pct [-1, 1] (how much mid price moved from start)
        self.observation_space: spaces.Box = spaces.Box(
            low=-2.0, high=2.0, shape=(5,), dtype=np.float32
        )

        self.reset_count: int = 0
        self.mid_price: float = mid_price
        self.arrival_price: float = mid_price
        self.remaining_volume: float = total_volume
        self.current_step: int = 0
        self.order_book: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        if self.mode == "historical" and self.historical_prices:
            # Sample a random starting index
            max_start = len(self.historical_prices) - self.total_steps
            if max_start > 0:
                start_idx = random.randint(0, max_start)
                self.historical_trajectory = self.historical_prices[start_idx : start_idx + self.total_steps]
            else:
                self.historical_trajectory = [self.starting_mid_price] * self.total_steps
            
            self.mid_price = self.historical_trajectory[0]
        elif self.mode == "stress":
            self.mid_price = self.starting_mid_price * random.uniform(0.90, 0.98)
        else:
            # Randomize start parameters slightly to improve generalization
            self.mid_price = self.starting_mid_price * random.uniform(0.95, 1.05)
            
        self.arrival_price = self.mid_price
        self.remaining_volume = self.total_volume
        self.current_step = 0
        
        # Simulate initial order book
        current_depth_scale = self.depth_scale
        current_spread = random.uniform(1.0, 5.0)
        if self.mode == "stress":
            current_depth_scale = self.depth_scale / 10.0
            current_spread = random.uniform(20.0, 100.0)
            
        self.order_book = OrderBookSimulator.generate_synthetic_order_book(
            mid_price=self.mid_price,
            spread=current_spread,
            depth_scale=current_depth_scale,
            noise_level=0.2
        )
        
        self.history = []
        self.reset_count += 1
        
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        remaining_vol_pct = self.remaining_volume / self.total_volume
        remaining_time_pct = (self.total_steps - self.current_step) / self.total_steps
        
        best_bid = float(self.order_book["bids"][0][0])
        best_ask = float(self.order_book["asks"][0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / self.mid_price) * 100.0
        
        # Calculate book imbalance (bid vs ask depth at level 1)
        bid_depth_1 = sum(v for _, v in self.order_book["bids"][:5])
        ask_depth_1 = sum(v for _, v in self.order_book["asks"][:5])
        imbalance = (bid_depth_1 - ask_depth_1) / (bid_depth_1 + ask_depth_1 + 1e-8)
        
        price_dev_pct = ((self.mid_price - self.arrival_price) / self.arrival_price) * 100.0
        
        # Scale for neural network input
        obs = np.array([
            remaining_vol_pct,
            remaining_time_pct,
            np.clip(spread_pct, 0.0, 1.0),
            np.clip(imbalance, -1.0, 1.0),
            np.clip(price_dev_pct / 5.0, -1.0, 1.0)  # scale typical 5% price moves
        ], dtype=np.float32)
        
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Action is a float inside [0, 1]
        action_val = float(np.clip(action[0], 0.0, 1.0))
        
        # On the last step, force execution of the remaining inventory
        is_last_step = (self.current_step == self.total_steps - 1)
        if is_last_step:
            sell_qty = self.remaining_volume
        else:
            sell_qty = self.remaining_volume * action_val
            
        # Ensure we don't over-sell
        sell_qty = min(sell_qty, self.remaining_volume)
        
        # Simulate execution
        slippage_pct = 0.0
        vwap = self.mid_price
        filled_qty = 0.0
        
        if sell_qty > 0:
            # Route percentage to OTC and rest to L2 Book
            otc_qty = sell_qty * (self.otc_pct / 100.0)
            market_qty = sell_qty - otc_qty
            
            total_value = 0.0
            best_bid = float(self.order_book["bids"][0][0]) if len(self.order_book["bids"]) > 0 else self.mid_price
            
            if otc_qty > 0:
                # OTC execution executes at a fixed 0.05% discount from the best bid
                vwap_otc = best_bid * (1.0 - 0.0005)
                total_value += otc_qty * vwap_otc
                filled_qty += otc_qty
                
            if market_qty > 0:
                sim = OrderBookSimulator.simulate_market_sell(self.order_book, market_qty)
                vwap_mkt = float(sim["vwap"])
                filled_mkt = float(sim["filled_amount"])
                total_value += filled_mkt * vwap_mkt
                filled_qty += filled_mkt
            
            vwap = total_value / filled_qty if filled_qty > 0 else self.mid_price
            slippage_pct = ((best_bid - vwap) / best_bid) * 100.0 if best_bid > 0 and vwap > 0 else 0.0
            
            # Reduce inventory
            self.remaining_volume -= filled_qty
        
        # Update market price
        if self.mode == "historical" and hasattr(self, "historical_trajectory"):
            next_idx = self.current_step + 1
            if next_idx < len(self.historical_trajectory):
                self.mid_price = self.historical_trajectory[next_idx]
            else:
                self.mid_price = self.historical_trajectory[-1]
        else:
            # Update market price (Random walk / GBM with drift from sales)
            # OTC transactions have lower direct market impact on standard exchanges (e.g. 10x lower impact)
            market_qty_effective = (sell_qty - sell_qty * (self.otc_pct / 100.0)) + 0.1 * (sell_qty * (self.otc_pct / 100.0))
            
            if self.mode == "stress":
                market_impact_impact = market_qty_effective * 0.0005 # 5x higher impact
                price_drift = -market_impact_impact + random.normalvariate(-self.mid_price * 0.005, self.mid_price * 0.002) # panic sell drift
            else:
                market_impact_impact = market_qty_effective * 0.0001
                price_drift = -market_impact_impact + random.normalvariate(0, self.mid_price * 0.001)
                
            self.mid_price = max(1000.0, self.mid_price + price_drift)
        
        # Generate new order book at the new price level
        current_depth_scale = self.depth_scale
        current_spread = random.uniform(1.0, 5.0)
        if self.mode == "stress":
            current_depth_scale = self.depth_scale / 10.0
            current_spread = random.uniform(20.0, 100.0)
            
        self.order_book = OrderBookSimulator.generate_synthetic_order_book(
            mid_price=self.mid_price,
            spread=current_spread,
            depth_scale=current_depth_scale,
            noise_level=0.2
        )
        
        # Calculate Reward
        execution_reward = 0.0
        if filled_qty > 0:
            execution_cost = (self.arrival_price - vwap) / self.arrival_price
            execution_reward = -execution_cost * (filled_qty / self.total_volume) * 100.0
            
        inventory_penalty = -0.05 * (self.remaining_volume / self.total_volume)
        
        shortfall_penalty = 0.0
        if is_last_step and self.remaining_volume > 0:
            shortfall_penalty = -5.0 * (self.remaining_volume / self.total_volume)
            
        reward = execution_reward + inventory_penalty + shortfall_penalty
        
        # Save history
        self.history.append({
            "step": self.current_step,
            "action": action_val,
            "sell_qty": sell_qty,
            "filled_qty": filled_qty,
            "vwap": vwap,
            "slippage_pct": slippage_pct,
            "remaining_volume": self.remaining_volume,
            "mid_price": self.mid_price
        })
        
        self.current_step += 1
        terminated = (self.remaining_volume <= 1e-4) or (self.current_step >= self.total_steps)
        truncated = False
        
        return self._get_observation(), float(reward), terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        if len(self.history) > 0:
            last = self.history[-1]
            logging.info(
                f"Step {last['step']}: Action={last['action']:.3f} | "
                f"Sold={last['filled_qty']:.2f} BTC | VWAP={last['vwap']:.2f} | "
                f"Remaining={last['remaining_volume']:.2f} BTC | Price={last['mid_price']:.2f}"
            )
