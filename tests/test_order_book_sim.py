import pytest
from src.order_book_sim import OrderBookSimulator

def test_generate_synthetic_order_book():
    mid = 60000.0
    spread = 10.0
    num_levels = 20
    
    book = OrderBookSimulator.generate_synthetic_order_book(
        mid_price=mid,
        spread=spread,
        num_levels=num_levels
    )
    
    assert len(book["bids"]) == num_levels
    assert len(book["asks"]) == num_levels
    
    # Check sorting
    bids = book["bids"]
    asks = book["asks"]
    
    # Bids descending order of price
    for i in range(len(bids) - 1):
        assert bids[i][0] > bids[i+1][0]
        
    # Asks ascending order of price
    for i in range(len(asks) - 1):
        assert asks[i][0] < asks[i+1][0]
        
    # Check spread bounds
    assert bids[0][0] <= mid - spread/2.0
    assert asks[0][0] >= mid + spread/2.0

def test_simulate_market_sell():
    book = {
        "bids": [
            [100.0, 2.0], # price, qty
            [90.0, 3.0],
            [80.0, 5.0]
        ],
        "asks": [
            [110.0, 5.0]
        ]
    }
    
    # Sell 1.0 unit (should execute entirely at 100.0)
    res = OrderBookSimulator.simulate_market_sell(book, 1.0)
    assert res["filled_amount"] == 1.0
    assert res["vwap"] == 100.0
    assert res["slippage_pct"] == 0.0
    
    # Sell 4.0 units (2.0 at 100.0, 2.0 at 90.0)
    # vwap = (2*100 + 2*90) / 4 = 95.0
    # best bid = 100.0
    # slippage = (100 - 95)/100 = 5.0%
    res2 = OrderBookSimulator.simulate_market_sell(book, 4.0)
    assert res2["filled_amount"] == 4.0
    assert res2["vwap"] == 95.0
    assert abs(res2["slippage_pct"] - 5.0) < 1e-4
    assert res2["marginal_price"] == 90.0
    assert abs(res2["price_impact_pct"] - 10.0) < 1e-4
