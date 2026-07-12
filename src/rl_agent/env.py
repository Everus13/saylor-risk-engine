import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
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
        depth_scale: float = 12.0     # L2 book depth scale (liquidity)
    ):
        super().__init__()
        self.total_volume = total_volume
        self.total_steps = total_steps
        self.starting_mid_price = mid_price
        self.depth_scale = depth_scale

        # Action Space: continuous value in [0.0, 1.0] representing
        # the fraction of REMAINING volume to sell in this step.
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space:
        # 1. remaining_volume_pct [0, 1]
        # 2. remaining_time_pct [0, 1]
        # 3. spread_pct [-1, 1] (rescaled)
        # 4. book_imbalance [-1, 1]
        # 5. price_deviation_pct [-1, 1] (how much mid price moved from start)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(5,), dtype=np.float32
        )

        self.reset_count = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize start parameters slightly to improve generalization
        self.mid_price = self.starting_mid_price * random.uniform(0.95, 1.05)
        self.arrival_price = self.mid_price
        self.remaining_volume = self.total_volume
        self.current_step = 0
        
        # Simulate initial order book
        self.order_book = OrderBookSimulator.generate_synthetic_order_book(
            mid_price=self.mid_price,
            spread=random.uniform(1.0, 5.0),
            depth_scale=self.depth_scale,
            noise_level=0.2
        )
        
        self.history = []
        self.reset_count += 1
        
        return self._get_observation(), {}

    def _get_observation(self):
        remaining_vol_pct = self.remaining_volume / self.total_volume
        remaining_time_pct = (self.total_steps - self.current_step) / self.total_steps
        
        best_bid = self.order_book["bids"][0][0]
        best_ask = self.order_book["asks"][0][0]
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

    def step(self, action):
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
            sim = OrderBookSimulator.simulate_market_sell(self.order_book, sell_qty)
            vwap = sim["vwap"]
            slippage_pct = sim["slippage_pct"]
            filled_qty = sim["filled_amount"]
            
            # Reduce inventory
            self.remaining_volume -= filled_qty
        
        # Update market price (Random walk / GBM with drift from sales)
        # Sell orders push the price down (market impact)
        market_impact_impact = sell_qty * 0.0001  # simple price impact multiplier
        price_drift = -market_impact_impact + random.normalvariate(0, self.mid_price * 0.001)
        self.mid_price = max(1000.0, self.mid_price + price_drift)
        
        # Generate new order book at the new price level
        self.order_book = OrderBookSimulator.generate_synthetic_order_book(
            mid_price=self.mid_price,
            spread=random.uniform(1.0, 5.0),
            depth_scale=self.depth_scale,
            noise_level=0.2
        )
        
        # Calculate Reward
        # We penalize slippage (deviation from mid_price / arrival_price)
        # We also penalize holding inventory to mitigate risk (price drop)
        execution_reward = 0.0
        if filled_qty > 0:
            # Slippage cost term (relative to arrival price)
            execution_cost = (self.arrival_price - vwap) / self.arrival_price
            # Reward: minimize cost
            execution_reward = -execution_cost * (filled_qty / self.total_volume) * 100.0
            
        # Inventory penalty (holding risk)
        inventory_penalty = -0.05 * (self.remaining_volume / self.total_volume)
        
        # Penalty for failing to sell everything (should be 0 since we force-execute on last step,
        # but just in case of volume constraints):
        shortfall_penalty = 0.0
        if is_last_step and self.remaining_volume > 0:
            shortfall_penalty = -5.0 * (self.remaining_volume / self.total_volume)
            
        reward = execution_reward + inventory_penalty + shortfall_penalty
        
        # Save history for rendering/logging
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

    def render(self, mode="human"):
        if len(self.history) > 0:
            last = self.history[-1]
            print(f"Step {last['step']}: Action={last['action']:.3f} | Sold={last['filled_qty']:.2f} BTC | VWAP={last['vwap']:.2f} | Remaining={last['remaining_volume']:.2f} BTC | Price={last['mid_price']:.2f}")
