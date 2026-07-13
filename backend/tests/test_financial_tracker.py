import pytest
import datetime
from src.financial_tracker import FinancialTracker

def test_financial_tracker_initialization():
    tracker = FinancialTracker()
    assert tracker.data is not None
    assert "usd_cash_reserve" in tracker.data
    assert "convertible_notes" in tracker.data

def test_calculate_obligations():
    tracker = FinancialTracker()
    # Test over 30 days
    start_date = datetime.date(2026, 3, 1)
    res = tracker.calculate_obligations(start_date, 30)
    
    assert "total_usd_required" in res
    assert "total_opex_usd" in res
    
    # Verify that opex is zero
    assert res["total_opex_usd"] == 0.0

def test_btc_sell_requirements_math():
    tracker = FinancialTracker()
    # Override reserve and run requirements calculation
    tracker.update_cash_reserve(1000000.0) # 1M USD
    
    reqs = tracker.get_btc_sell_requirements(days_forecast=30, btc_price=50000.0)
    
    assert reqs["btc_price"] == 50000.0
    assert reqs["usd_cash_reserve"] == 1000000.0
    
    # Net USD needed should be max(0, total_required - 1M)
    expected_net = max(0.0, reqs["total_usd_required"] - 1000000.0)
    assert abs(reqs["net_usd_needed"] - expected_net) < 1e-4
    
    # BTC to sell should be net_usd / price
    expected_btc = expected_net / 50000.0
    assert abs(reqs["btc_to_sell_stress_case"] - expected_btc) < 1e-4

def test_fetch_sec_edgar_facts():
    tracker = FinancialTracker()
    facts = tracker.fetch_sec_edgar_facts()
    
    assert "long_term_debt" in facts
    assert "long_term_debt_current" in facts
    assert "date" in facts
    assert facts["long_term_debt"] > 0

def test_sync_obligations_with_sec():
    tracker = FinancialTracker()
    start_date = datetime.date(2026, 6, 1)
    
    # Run with SEC sync disabled
    res_no_sync = tracker.sync_obligations_with_sec(start_date, 30, use_sec_debt=False)
    
    # Run with SEC sync enabled
    res_sync = tracker.sync_obligations_with_sec(start_date, 30, use_sec_debt=True)
    
    assert res_sync["total_usd_required"] >= res_no_sync["total_usd_required"]
    
    # Run dynamic sell reqs
    reqs = tracker.get_btc_sell_requirements_sync(days_forecast=30, btc_price=60000.0, use_sec_debt=True)
    res_today = tracker.sync_obligations_with_sec(datetime.date.today(), 30, use_sec_debt=True)
    assert reqs["btc_price"] == 60000.0
    assert reqs["total_usd_required"] == res_today["total_usd_required"]


