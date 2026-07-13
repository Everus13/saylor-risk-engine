import numpy as np
import pandas as pd
import time
import random
import logging
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    import ccxt
except ImportError:
    ccxt = None

class OrderBookSimulator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_synthetic_order_book(
        mid_price: float = 60000.0,
        spread: float = 2.0,
        num_levels: int = 50,
        depth_scale: float = 10.0,  # Average volume per level in BTC
        noise_level: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generate a synthetic L2 order book (bids and asks) with realistic exponential decay of volume
        and random micro-fluctuations.
        """
        bids: List[List[float]] = []
        asks: List[List[float]] = []
        
        best_bid = mid_price - spread / 2.0
        best_ask = mid_price + spread / 2.0
        
        # Step size per level in percentage
        price_step_pct = 0.0002  # 0.02%
        
        for i in range(num_levels):
            # Price calculations
            bid_price = best_bid * (1.0 - i * price_step_pct)
            ask_price = best_ask * (1.0 + i * price_step_pct)
            
            # Volume calculations (exponential decay with depth, plus noise)
            decay_factor = np.exp(-i * 0.03)
            
            bid_noise = 1.0 + random.uniform(-noise_level, noise_level)
            ask_noise = 1.0 + random.uniform(-noise_level, noise_level)
            
            bid_volume = depth_scale * decay_factor * bid_noise
            ask_volume = depth_scale * decay_factor * ask_noise
            
            # Clamp volume to be positive
            bid_volume = max(0.01, bid_volume)
            ask_volume = max(0.01, ask_volume)
            
            bids.append([bid_price, bid_volume])
            asks.append([ask_price, ask_volume])
            
        return {
            "bids": bids,  # sorted descending by price
            "asks": asks,  # sorted ascending by price
            "timestamp": time.time()
        }

    @staticmethod
    def simulate_market_sell(order_book: Dict[str, Any], sell_amount: float) -> Dict[str, Any]:
        """
        Simulate selling BTC into the bid side of the order book using vectorized NumPy operations.
        """
        bids_list = order_book.get("bids", [])
        if not bids_list or sell_amount <= 0:
            return {
                "requested_amount": sell_amount,
                "filled_amount": 0.0,
                "unfilled_amount": sell_amount,
                "vwap": 0.0,
                "best_bid": 0.0,
                "marginal_price": 0.0,
                "slippage_pct": 0.0,
                "price_impact_pct": 0.0,
                "steps": []
            }

        bids = np.array(bids_list, dtype=np.float64)
        prices = bids[:, 0]
        volumes = bids[:, 1]
        
        cum_volumes = np.cumsum(volumes)
        total_volume = cum_volumes[-1]
        best_bid = prices[0]
        
        if sell_amount >= total_volume:
            # Eat all bids in the book
            filled_qty = total_volume
            unfilled = sell_amount - total_volume
            total_value = np.sum(prices * volumes)
            marginal_price = prices[-1]
            steps = bids.tolist()
        else:
            # Find the first level index where cumulative volume satisfies the order size
            idx = int(np.searchsorted(cum_volumes, sell_amount))
            
            # Sum up fully filled levels prior to index
            if idx > 0:
                fully_filled_value = np.sum(prices[:idx] * volumes[:idx])
                fully_filled_volume = cum_volumes[idx - 1]
            else:
                fully_filled_value = 0.0
                fully_filled_volume = 0.0
                
            # Handle partial fill at the boundary index
            partial_fill_qty = sell_amount - fully_filled_volume
            partial_fill_value = partial_fill_qty * prices[idx]
            
            total_value = fully_filled_value + partial_fill_value
            filled_qty = sell_amount
            unfilled = 0.0
            marginal_price = prices[idx]
            
            # Format steps output
            steps = bids[:idx].tolist()
            steps.append([prices[idx], partial_fill_qty])
            
        vwap = total_value / filled_qty if filled_qty > 0 else 0.0
        slippage_pct = ((best_bid - vwap) / best_bid) * 100.0 if best_bid > 0 and vwap > 0 else 0.0
        price_impact_pct = ((best_bid - marginal_price) / best_bid) * 100.0 if best_bid > 0 else 0.0
        
        return {
            "requested_amount": sell_amount,
            "filled_amount": filled_qty,
            "unfilled_amount": unfilled,
            "vwap": vwap,
            "best_bid": best_bid,
            "marginal_price": marginal_price,
            "slippage_pct": slippage_pct,
            "price_impact_pct": price_impact_pct,
            "steps": steps
        }

    @staticmethod
    def get_live_order_book(exchange_id: str = "binance", symbol: str = "BTC/USDT") -> Optional[Dict[str, Any]]:
        """
        Fetch L2 order book in real-time from a cryptocurrency exchange via ccxt.
        Returns a structured dict with bids and asks.
        """
        if ccxt is None:
            logging.error("ccxt library is not installed. Live order book unavailable.")
            raise ImportError("ccxt library is not installed.")
            
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class()
            limit = 100
            book = exchange.fetch_order_book(symbol, limit=limit)
            return {
                "bids": book["bids"],
                "asks": book["asks"],
                "timestamp": book["timestamp"] / 1000.0 if book["timestamp"] else time.time()
            }
        except Exception as e:
            logging.error(f"Error fetching live order book from {exchange_id} for {symbol}: {e}")
            return None
