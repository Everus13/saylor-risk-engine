import numpy as np
import pandas as pd
import time
import random
try:
    import ccxt
except ImportError:
    ccxt = None

class OrderBookSimulator:
    def __init__(self):
        pass

    @staticmethod
    def generate_synthetic_order_book(
        mid_price: float = 60000.0,
        spread: float = 2.0,
        num_levels: int = 50,
        depth_scale: float = 10.0,  # Average volume per level in BTC
        noise_level: float = 0.2
    ):
        """
        Generate a synthetic L2 order book (bids and asks) with realistic exponential decay of volume
        and random micro-fluctuations.
        """
        bids = []
        asks = []
        
        best_bid = mid_price - spread / 2.0
        best_ask = mid_price + spread / 2.0
        
        # Step size per level in percentage
        price_step_pct = 0.0002  # 0.02%
        
        for i in range(num_levels):
            # Price calculations
            bid_price = best_bid * (1.0 - i * price_step_pct)
            ask_price = best_ask * (1.0 + i * price_step_pct)
            
            # Volume calculations (exponential decay with depth, plus noise)
            # Volume is higher near key levels, but decays on average as we go deep
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
    def simulate_market_sell(order_book: dict, sell_amount: float):
        """
        Simulate selling BTC into the bid side of the order book.
        Calculates:
          - filled_volume: actual volume filled (may be less than sell_amount if order book is thin)
          - vwap: volume-weighted average price of execution
          - slippage_pct: percentage difference between best bid and vwap
          - price_impact_pct: percentage difference between best bid and final executed marginal bid price
          - execution_steps: list of [price, filled_qty]
        """
        bids = order_book["bids"]
        remaining = sell_amount
        total_value = 0.0
        filled_qty = 0.0
        execution_steps = []
        best_bid = bids[0][0] if len(bids) > 0 else 0.0
        final_price = best_bid
        
        for price, volume in bids:
            if remaining <= 0:
                break
            
            fill = min(remaining, volume)
            total_value += fill * price
            filled_qty += fill
            remaining -= fill
            final_price = price
            execution_steps.append([price, fill])
            
        vwap = total_value / filled_qty if filled_qty > 0 else 0.0
        slippage_pct = ((best_bid - vwap) / best_bid) * 100.0 if best_bid > 0 and vwap > 0 else 0.0
        price_impact_pct = ((best_bid - final_price) / best_bid) * 100.0 if best_bid > 0 else 0.0
        
        return {
            "requested_amount": sell_amount,
            "filled_amount": filled_qty,
            "unfilled_amount": remaining,
            "vwap": vwap,
            "best_bid": best_bid,
            "marginal_price": final_price,
            "slippage_pct": slippage_pct,
            "price_impact_pct": price_impact_pct,
            "steps": execution_steps
        }

    @staticmethod
    def get_live_order_book(exchange_id: str = "binance", symbol: str = "BTC/USDT"):
        """
        Fetch L2 order book in real-time from a cryptocurrency exchange via ccxt.
        Returns a structured dict with bids and asks.
        """
        if ccxt is None:
            raise ImportError("ccxt library is not installed.")
            
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class()
            # Fetch L2 book with limit of 100 levels
            limit = 100
            book = exchange.fetch_order_book(symbol, limit=limit)
            return {
                "bids": book["bids"],  # [ [price, volume], ... ]
                "asks": book["asks"],
                "timestamp": book["timestamp"] / 1000.0 if book["timestamp"] else time.time()
            }
        except Exception as e:
            print(f"Error fetching live order book from {exchange_id} for {symbol}: {e}")
            # Fallback to generating synthetic book
            return None
