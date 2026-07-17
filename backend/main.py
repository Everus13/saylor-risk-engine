import os
import sys
import datetime
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure the root folder and backend folder are in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.src.financial_tracker import FinancialTracker
from backend.src.order_book_sim import OrderBookSimulator
from backend.src.ml_impact import PriceImpactMLPipeline
from backend.src.rl_agent.train import RLExecutionTrainer, SB3_AVAILABLE
from backend.src import database

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="MSTR-BTC Impact & Execution Engine API")

# Configure CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Business Logic
tracker = FinancialTracker()
ml_pipeline = PriceImpactMLPipeline()
rl_trainer = RLExecutionTrainer()

# Ensure model weights are loaded
ml_pipeline.load_models()

# --- Pydantic Request Models ---

class SettingsUpdate(BaseModel):
    usd_cash_reserve: float

class SimulationRequest(BaseModel):
    sell_qty: float
    exchange_select: str  # "synthetic" or "live"
    seed: Optional[int] = None

class RLSimulationRequest(BaseModel):
    total_volume: float
    total_steps: int
    depth_scale: float
    otc_pct: float
    strategy: str  # "rl" or "twap"
    seed: Optional[int] = None
    agent_type: Optional[str] = "standard"

class SettingsMNAVUpdate(BaseModel):
    btc_holdings: float
    shares_outstanding: float

# --- WebSocket Log Handler for Streaming RL Logs ---

class WebSocketLogHandler(logging.Handler):
    def __init__(self, websockets: List[WebSocket]):
        super().__init__()
        self.websockets = websockets

    def emit(self, record):
        log_entry = self.format(record)
        for ws in self.websockets:
            try:
                # Use asyncio run_coroutine_threadsafe to send async from logger sync context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(ws.send_text(log_entry))
            except Exception:
                pass

# Active WebSockets for training logs
active_training_websockets: List[WebSocket] = []

# --- API Endpoints ---

@app.get("/api/prices")
def get_prices():
    """Fetch live prices from tracker."""
    try:
        prices = tracker.get_current_prices()
        return prices
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/financials")
def get_financials(forecast_days: int = 90, use_sec_debt: bool = True):
    """Fetch dynamic obligations, cash reserves, and net BTC sell requirements."""
    try:
        live_prices = tracker.get_current_prices()
        btc_price = live_prices["BTC"]
        
        reqs = tracker.get_btc_sell_requirements_sync(
            days_forecast=forecast_days,
            btc_price=btc_price,
            use_sec_debt=use_sec_debt
        )
        
        # Calculate PyTorch model estimated price drop for the required volume
        if reqs["btc_to_sell_stress_case"] > 0:
            book_temp = OrderBookSimulator.generate_synthetic_order_book(mid_price=btc_price)
            torch_preds = ml_pipeline.predict_impact(book_temp, reqs["btc_to_sell_stress_case"], model_type="pytorch")
            impact_val = torch_preds[0.5]
            impact_worst = torch_preds[0.9]
        else:
            impact_val = 0.0
            impact_worst = 0.0
            
        sec_facts = tracker.fetch_sec_edgar_facts()
        
        # Convert date objects to string for JSON serialization
        payments = reqs["breakdown"]["payments_breakdown"]
        for p in payments:
            if isinstance(p["date"], (datetime.date, datetime.datetime)):
                p["date"] = p["date"].strftime("%Y-%m-%d")

        return {
            "usd_cash_reserve": reqs["usd_cash_reserve"],
            "total_usd_required": reqs["total_usd_required"],
            "net_usd_needed": reqs["net_usd_needed"],
            "btc_price": reqs["btc_price"],
            "btc_to_sell_stress_case": reqs["btc_to_sell_stress_case"],
            "pytorch_impact_pct": impact_val,
            "pytorch_impact_worst_pct": impact_worst,
            "sec_facts": sec_facts,
            "payments_breakdown": payments
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings")
def update_settings(data: SettingsUpdate):
    """Update USD cash reserve in memory and database."""
    try:
        tracker.update_cash_reserve(data.usd_cash_reserve)
        return {"status": "success", "usd_cash_reserve": data.usd_cash_reserve}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate/impact")
def simulate_impact(data: SimulationRequest):
    """Simulate L2 order book walk and predict price impact using ML models."""
    try:
        if data.seed is not None:
            import random
            import numpy as np
            random.seed(data.seed)
            np.random.seed(data.seed)
        live_prices = tracker.get_current_prices()
        btc_price = live_prices["BTC"]
        
        # Get order book
        if data.exchange_select == "live":
            book = OrderBookSimulator.get_live_order_book("binance", "BTC/USDT")
            if book is None:
                book = OrderBookSimulator.generate_synthetic_order_book(mid_price=btc_price)
        else:
            book = OrderBookSimulator.generate_synthetic_order_book(mid_price=btc_price)
            
        # Simulate
        sim_res = OrderBookSimulator.simulate_market_sell(book, data.sell_qty)
        
        # ML Predictions
        torch_preds = ml_pipeline.predict_impact(book, data.sell_qty, model_type="pytorch")
        
        # Save to DB
        database.add_simulation_record(
            strategy="l2_walk",
            sell_volume=data.sell_qty,
            steps=1,
            avg_price=sim_res["vwap"],
            total_revenue=sim_res["filled_amount"] * sim_res["vwap"],
            slippage_pct=sim_res["slippage_pct"],
            details=[{
                "step": 0,
                "action": 1.0,
                "sell_qty": data.sell_qty,
                "filled_qty": sim_res["filled_amount"],
                "vwap": sim_res["vwap"],
                "slippage_pct": sim_res["slippage_pct"],
                "remaining_volume": sim_res["unfilled_amount"],
                "mid_price": sim_res["best_bid"]
            }]
        )
        
        return {
            "slippage_pct": sim_res["slippage_pct"],
            "vwap": sim_res["vwap"],
            "marginal_price": sim_res["marginal_price"],
            "price_impact_pct": sim_res["price_impact_pct"],
            "order_book": {
                "bids": book["bids"][:50],  # Limit to 50 levels for response size
                "asks": book["asks"][:50]
            },
            "predictions": {
                "lgb": {0.1: torch_preds[0.1], 0.5: torch_preds[0.5], 0.9: torch_preds[0.9]},
                "pytorch": {0.1: torch_preds[0.1], 0.5: torch_preds[0.5], 0.9: torch_preds[0.9]}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate/rl")
def simulate_rl(data: RLSimulationRequest):
    """Run execution simulation using RL Agent vs TWAP."""
    try:
        # Fetch current database configuration for holdings & shares
        from backend.src.config import MSTR_BTC_HOLDINGS_DEFAULT, MSTR_SHARES_OUTSTANDING_DEFAULT
        btc_holdings = float(database.get_setting("btc_holdings", str(MSTR_BTC_HOLDINGS_DEFAULT)))
        shares_outstanding = float(database.get_setting("shares_outstanding", str(MSTR_SHARES_OUTSTANDING_DEFAULT)))
        cash_reserve = float(database.get_setting("usd_cash_reserve", str(3000000000.0)))

        sec_facts = tracker.fetch_sec_edgar_facts()
        long_term_debt = float(sec_facts.get("long_term_debt", 8196524000.0))

        live_prices = tracker.get_current_prices()
        btc_price = live_prices["BTC"]
        mstr_price = live_prices["MSTR"]

        headers = {"User-Agent": "Mozilla/5.0"}
        vix_price = 14.5
        dxy_price = 104.0
        tnx_price = 4.2
        try:
            import requests
            res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=1d", headers=headers, timeout=2)
            if res.status_code == 200:
                vix_price = float(res.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
            res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?interval=1d&range=1d", headers=headers, timeout=2)
            if res.status_code == 200:
                dxy_price = float(res.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
            res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?interval=1d&range=1d", headers=headers, timeout=2)
            if res.status_code == 200:
                tnx_price = float(res.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
        except Exception:
            pass

        # Calculate probability of mNAV collapse
        from backend.src.mnav_model import MNAVModelPipeline
        mnav_pipeline = MNAVModelPipeline()
        prob, _ = mnav_pipeline.predict_collapse_probability(
            current_btc=btc_price,
            current_mstr=mstr_price,
            current_vix=vix_price,
            current_dxy=dxy_price,
            current_tnx=tnx_price,
            btc_holdings=btc_holdings,
            shares_outstanding=shares_outstanding,
            cash_reserve=cash_reserve,
            long_term_debt=long_term_debt
        )

        # Calculate balance-sheet Liquidity Urgency factor to scale forced sale probability
        short_term_debt = float(sec_facts.get("long_term_debt_current", 0.0))
        if short_term_debt <= 0:
            short_term_debt = long_term_debt * 0.02
            
        shortfall = max(0.0, short_term_debt - cash_reserve)
        
        import math
        if shortfall <= 0:
            liquidity_urgency = 0.15
        else:
            ratio = shortfall / (shortfall + cash_reserve)
            liquidity_urgency = 0.15 + 0.70 * ratio
            
        forced_sale_prob = float(prob * liquidity_urgency)

        # Apply risk multiplier
        effective_volume = data.total_volume * (1.0 + forced_sale_prob)

        # Run selected strategy simulation
        df_strat, metrics_strat = rl_trainer.run_simulation(
            total_volume=effective_volume,
            total_steps=data.total_steps,
            starting_mid_price=btc_price,
            strategy=data.strategy,
            depth_scale=data.depth_scale,
            otc_pct=data.otc_pct,
            seed=data.seed,
            agent_type=data.agent_type
        )
        
        # Run TWAP simulation for comparison
        df_twap, metrics_twap = rl_trainer.run_simulation(
            total_volume=effective_volume,
            total_steps=data.total_steps,
            starting_mid_price=metrics_strat["arrival_price"],
            strategy="twap",
            depth_scale=data.depth_scale,
            otc_pct=data.otc_pct,
            seed=data.seed,
            agent_type=data.agent_type
        )
        
        # Convert DataFrames to dict lists
        strat_steps = df_strat.to_dict(orient="records")
        twap_steps = df_twap.to_dict(orient="records")
        
        # Save execution record to DB
        database.add_simulation_record(
            strategy=data.strategy,
            sell_volume=data.total_volume,
            steps=data.total_steps,
            avg_price=metrics_strat["avg_execution_price"],
            total_revenue=metrics_strat["total_revenue_usd"],
            slippage_pct=metrics_strat["total_slippage_pct"],
            details=strat_steps
        )
        
        return {
            "metrics": metrics_strat,
            "steps": strat_steps,
            "twap_metrics": metrics_twap,
            "twap_steps": twap_steps
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
def get_history(limit: int = 15):
    """Retrieve execution history from database."""
    try:
        history = database.get_simulation_history(limit=limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/status")
def get_market_status():
    """Evaluate 24h market performance to detect bearish shock phase."""
    try:
        live_prices = tracker.get_current_prices()
        btc_price = live_prices["BTC"]
        
        from src.rl_agent.train import fetch_historical_btc_prices
        hist_prices = fetch_historical_btc_prices()
        
        if hist_prices and len(hist_prices) > 0:
            yesterday_price = hist_prices[-1]
            change_pct = ((btc_price - yesterday_price) / yesterday_price) * 100.0
        else:
            change_pct = 0.0
            
        status = "bearish" if change_pct <= -3.0 else "neutral"
        
        return {
            "status": status,
            "change_24h_pct": change_pct,
            "current_price": btc_price
        }
    except Exception as e:
        return {
            "status": "neutral",
            "change_24h_pct": 0.0,
            "error": str(e)
        }

# --- WebSockets Endpoint for Training Stream ---

@app.websocket("/ws/rl/train")
async def websocket_rl_train(websocket: WebSocket):
    """WebSocket connection that starts training RL agent and streams logs."""
    await websocket.accept()
    active_training_websockets.append(websocket)
    
    # Add custom logger handler to stream logs to ws
    ws_handler = WebSocketLogHandler(active_training_websockets)
    ws_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
    logger = logging.getLogger()
    logger.addHandler(ws_handler)
    
    try:
        # Receive parameters
        data = await websocket.receive_json()
        timesteps = int(data.get("timesteps", 10000))
        volume = float(data.get("volume", 300.0))
        steps = int(data.get("steps", 15))
        depth_scale = float(data.get("depth_scale", 12.0))
        otc_pct = float(data.get("otc_pct", 0.0))
        agent_type = data.get("agent_type", "standard")
        
        await websocket.send_text(f"🚀 Инициализация Gymnasium-среды для обучения ({agent_type})...")
        
        # Run training in block executor to keep event loop free
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            lambda: rl_trainer.train_agent(
                total_timesteps=timesteps,
                total_volume=volume,
                total_steps=steps,
                mid_price=tracker.get_current_prices()["BTC"],
                depth_scale=depth_scale,
                otc_pct=otc_pct,
                agent_type=agent_type
            )
        )
        
        await websocket.send_text("✅ Обучение RL PPO-агента успешно завершено!")
    except WebSocketDisconnect:
        logging.info("WebSocket client disconnected from training channel.")
    except Exception as e:
        await websocket.send_text(f"❌ Ошибка в процессе обучения: {str(e)}")
    finally:
        logger.removeHandler(ws_handler)
        if websocket in active_training_websockets:
            active_training_websockets.remove(websocket)
        try:
            await websocket.close()
        except Exception:
            pass

# Run training trigger endpoint (optional HTTP fallback)
@app.post("/api/rl/train")
async def trigger_rl_train(timesteps: int = 10000, volume: float = 300.0, steps: int = 15, depth_scale: float = 12.0, otc_pct: float = 0.0, agent_type: str = "standard"):
    if not SB3_AVAILABLE:
        raise HTTPException(status_code=400, detail="stable-baselines3 is not installed.")
    try:
        btc_price = tracker.get_current_prices()["BTC"]
        # run training asynchronously
        asyncio.create_task(
            asyncio.to_thread(
                rl_trainer.train_agent,
                total_timesteps=timesteps,
                total_volume=volume,
                total_steps=steps,
                mid_price=btc_price,
                depth_scale=depth_scale,
                otc_pct=otc_pct,
                agent_type=agent_type
            )
        )
        return {"status": "success", "detail": f"Training of {agent_type} started in background thread."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings/mnav")
def update_mnav_settings(data: SettingsMNAVUpdate):
    try:
        database.set_setting("btc_holdings", str(data.btc_holdings))
        database.set_setting("shares_outstanding", str(data.shares_outstanding))
        return {"status": "success", "btc_holdings": data.btc_holdings, "shares_outstanding": data.shares_outstanding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mnav/status")
def get_mnav_status():
    try:
        from backend.src.config import MSTR_BTC_HOLDINGS_DEFAULT, MSTR_SHARES_OUTSTANDING_DEFAULT
        btc_holdings = float(database.get_setting("btc_holdings", str(MSTR_BTC_HOLDINGS_DEFAULT)))
        shares_outstanding = float(database.get_setting("shares_outstanding", str(MSTR_SHARES_OUTSTANDING_DEFAULT)))
        cash_reserve = float(database.get_setting("usd_cash_reserve", str(3000000000.0)))
        
        sec_facts = tracker.fetch_sec_edgar_facts()
        long_term_debt = float(sec_facts.get("long_term_debt", 8196524000.0))
        
        live_prices = tracker.get_current_prices()
        btc_price = live_prices["BTC"]
        mstr_price = live_prices["MSTR"]
        
        headers = {"User-Agent": "Mozilla/5.0"}
        vix_price = 14.5
        dxy_price = 104.0
        tnx_price = 4.2
        try:
            import requests
            res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX?interval=1d&range=1d", headers=headers, timeout=2)
            if res.status_code == 200:
                vix_price = float(res.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
            res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?interval=1d&range=1d", headers=headers, timeout=2)
            if res.status_code == 200:
                dxy_price = float(res.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
            res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX?interval=1d&range=1d", headers=headers, timeout=2)
            if res.status_code == 200:
                tnx_price = float(res.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])
        except Exception:
            pass
            
        from backend.src.mnav_model import MNAVModelPipeline
        pipeline = MNAVModelPipeline()
        prob, feats = pipeline.predict_collapse_probability(
            current_btc=btc_price,
            current_mstr=mstr_price,
            current_vix=vix_price,
            current_dxy=dxy_price,
            current_tnx=tnx_price,
            btc_holdings=btc_holdings,
            shares_outstanding=shares_outstanding,
            cash_reserve=cash_reserve,
            long_term_debt=long_term_debt
        )
        
        # Calculate balance-sheet Liquidity Urgency factor to scale forced sale probability
        short_term_debt = float(sec_facts.get("long_term_debt_current", 0.0))
        if short_term_debt <= 0:
            # Fallback baseline: 2% of total long term debt is due soon
            short_term_debt = long_term_debt * 0.02
            
        shortfall = max(0.0, short_term_debt - cash_reserve)
        
        import math
        if shortfall <= 0:
            # Low urgency if cash covers short-term debt
            liquidity_urgency = 0.15
        else:
            ratio = shortfall / (shortfall + cash_reserve)
            # Urgency scales between 15% and 85%
            liquidity_urgency = 0.15 + 0.70 * ratio
            
        forced_sale_prob = float(prob * liquidity_urgency)
        
        pipeline.load_model()
        model_name = getattr(pipeline, "model_name", "Random Forest")
        model_metrics = getattr(pipeline, "metrics", {})
        
        return {
            "btc_holdings": btc_holdings,
            "shares_outstanding": shares_outstanding,
            "usd_cash_reserve": cash_reserve,
            "long_term_debt": long_term_debt,
            "btc_price": btc_price,
            "mstr_price": mstr_price,
            "implied_btc_per_share": feats["implied_btc_per_share"],
            "mnav_per_share": feats["mnav_per_share"],
            "premium_pct": feats["premium_pct"],
            "collapse_probability": forced_sale_prob,  # output scaled realistic probability
            "vix": vix_price,
            "dxy": dxy_price,
            "tnx": tnx_price,
            "model_name": model_name,
            "model_metrics": model_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mnav/train")
def train_mnav_model():
    try:
        from backend.src.config import MSTR_BTC_HOLDINGS_DEFAULT, MSTR_SHARES_OUTSTANDING_DEFAULT
        btc_holdings = float(database.get_setting("btc_holdings", str(MSTR_BTC_HOLDINGS_DEFAULT)))
        shares_outstanding = float(database.get_setting("shares_outstanding", str(MSTR_SHARES_OUTSTANDING_DEFAULT)))
        cash_reserve = float(database.get_setting("usd_cash_reserve", str(3000000000.0)))
        sec_facts = tracker.fetch_sec_edgar_facts()
        long_term_debt = float(sec_facts.get("long_term_debt", 8196524000.0))
        
        from backend.src.mnav_model import MNAVModelPipeline
        pipeline = MNAVModelPipeline()
        res = pipeline.train_pipeline(
            btc_holdings=btc_holdings,
            shares_outstanding=shares_outstanding,
            cash_reserve=cash_reserve,
            long_term_debt=long_term_debt
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
