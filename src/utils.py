import os
import joblib
import torch
import numpy as np
import plotly.graph_objects as go
from src.config import LIGHTGBM_MODEL_PATH, PYTORCH_MODEL_PATH
from src.ml_impact import PriceImpactMLPipeline, generate_ml_dataset

def ensure_models_exist():
    """
    Check if LightGBM and PyTorch models exist.
    If not, train them with default parameters on a synthetic dataset
    so the app runs smoothly from day one.
    """
    lgb_exists = os.path.exists(LIGHTGBM_MODEL_PATH)
    torch_exists = os.path.exists(PYTORCH_MODEL_PATH)
    
    if not lgb_exists or not torch_exists:
        print("Pretrained models not found. Training on synthetic dataset...")
        X, y = generate_ml_dataset(n_snapshots=120)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        pipeline = PriceImpactMLPipeline()
        
        if not lgb_exists:
            pipeline.train_lightgbm_quantiles(X_train, y_train, X_val, y_val)
        if not torch_exists:
            pipeline.train_pytorch_quantiles(X_train, y_train, X_val, y_val)
        print("Initialization training complete.")
    else:
        print("Found pretrained models in cache.")

def plot_order_book_chart(order_book: dict, sell_size: float = None):
    """
    Create a beautiful Plotly chart showing bid/ask depth, spread, and optionally
    an overlay of where a market sell order eats the bids.
    """
    bids = np.array(order_book["bids"])
    asks = np.array(order_book["asks"])
    
    # Sort for cumulative depth calculation
    # Bids are sorted descending. Cumulative sum of bid volume
    cum_bid_vol = np.cumsum(bids[:, 1])
    # Asks are sorted ascending. Cumulative sum of ask volume
    cum_ask_vol = np.cumsum(asks[:, 1])
    
    fig = go.Figure()
    
    # Bids (Green)
    fig.add_trace(go.Scatter(
        x=bids[:, 0],
        y=cum_bid_vol,
        fill='tozeroy',
        mode='lines',
        name='Bids (Buyers)',
        line=dict(color='rgba(46, 204, 113, 1.0)', width=2),
        fillcolor='rgba(46, 204, 113, 0.2)'
    ))
    
    # Asks (Red)
    fig.add_trace(go.Scatter(
        x=asks[:, 0],
        y=cum_ask_vol,
        fill='tozeroy',
        mode='lines',
        name='Asks (Sellers)',
        line=dict(color='rgba(231, 76, 60, 1.0)', width=2),
        fillcolor='rgba(231, 76, 60, 0.2)'
    ))
    
    # If sell size is provided, show where the bids will be consumed
    if sell_size is not None and sell_size > 0:
        # Find index where cum_bid_vol exceeds sell_size
        idx = np.where(cum_bid_vol >= sell_size)[0]
        if len(idx) > 0:
            impact_price = bids[idx[0], 0]
        else:
            impact_price = bids[-1, 0]
            
        fig.add_vline(
            x=impact_price, 
            line_dash="dash", 
            line_color="yellow", 
            annotation_text=f"Marginal Price: ${impact_price:,.2f}"
        )
        
        # Shade the consumed area
        fig.add_vrect(
            x0=impact_price,
            x1=bids[0, 0],
            fillcolor="yellow",
            opacity=0.15,
            layer="below",
            line_width=0,
            annotation_text="Aggressive Sell Impact Zone",
            annotation_position="top left"
        )
        
    fig.update_layout(
        title="Binance L2 Depth & Market Impact Visualization",
        xaxis_title="Price (USD)",
        yaxis_title="Cumulative Volume (BTC)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return fig
