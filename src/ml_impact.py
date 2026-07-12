import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import os
import joblib
import random
import matplotlib.pyplot as plt

from src.config import LIGHTGBM_MODEL_PATH, PYTORCH_MODEL_PATH, LGBM_GRID, PYTORCH_CONFIG
from src.order_book_sim import OrderBookSimulator

# --- Pinball Loss for Quantile Regression ---
def pinball_loss_np(y_true, y_pred, q):
    """Calculate Pinball Loss using numpy."""
    diff = y_true - y_pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, targets):
        """
        preds: Tensor of shape (batch_size, num_quantiles)
        targets: Tensor of shape (batch_size, 1)
        """
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            error = targets - preds[:, i:i+1]
            loss += torch.max(q * error, (q - 1) * error).mean()
        return loss

# --- PyTorch Quantile Regression MLP Model ---
class QuantileMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, output_dim=3):
        # hidden_dims is kept for backward compatibility in signature but ignored for this premium arch
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.network(x)

# --- Feature Engineering ---
def extract_features(order_book: dict, sell_size: float):
    """
    Extract microstructural features from an L2 order book snapshot and a prospective sell size.
    """
    bids = order_book["bids"]
    asks = order_book["asks"]
    
    best_bid = bids[0][0]
    best_ask = asks[0][0]
    spread = best_ask - best_bid
    
    # Calculate depth at levels
    # first 5 levels
    depth_bid_1 = sum(v for _, v in bids[:5])
    depth_ask_1 = sum(v for _, v in asks[:5])
    
    # levels 5 to 15
    depth_bid_2 = sum(v for _, v in bids[5:15])
    depth_ask_2 = sum(v for _, v in asks[5:15])
    
    # levels 15 to 50
    depth_bid_3 = sum(v for _, v in bids[15:50])
    depth_ask_3 = sum(v for _, v in asks[15:50])
    
    total_bid_depth = depth_bid_1 + depth_bid_2 + depth_bid_3
    total_ask_depth = depth_ask_1 + depth_ask_2 + depth_ask_3
    
    # Book Imbalance
    imbalance_1 = (depth_bid_1 - depth_ask_1) / (depth_bid_1 + depth_ask_1 + 1e-8)
    imbalance_total = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth + 1e-8)
    
    # Order size ratios
    size_to_bid_1 = sell_size / (depth_bid_1 + 1e-8)
    size_to_total_bid = sell_size / (total_bid_depth + 1e-8)
    
    features = [
        best_bid,
        spread,
        spread / best_bid,
        depth_bid_1,
        depth_bid_2,
        depth_bid_3,
        depth_ask_1,
        depth_ask_2,
        depth_ask_3,
        imbalance_1,
        imbalance_total,
        size_to_bid_1,
        size_to_total_bid,
        sell_size
    ]
    return np.array(features, dtype=np.float32)

# --- Dataset Generator ---
def generate_ml_dataset(n_snapshots=200, levels=50):
    """
    Generate a dataset of features and targets using synthetic order books
    and market sell simulations.
    """
    data = []
    
    for _ in range(n_snapshots):
        # Vary market conditions
        mid_price = random.uniform(50000.0, 75000.0)
        spread = random.uniform(1.0, 10.0)
        depth_scale = random.uniform(5.0, 25.0)
        
        book = OrderBookSimulator.generate_synthetic_order_book(
            mid_price=mid_price,
            spread=spread,
            num_levels=levels,
            depth_scale=depth_scale,
            noise_level=0.3
        )
        
        # Sample different sell sizes
        for _ in range(5):
            sell_size = random.uniform(1.0, 500.0)
            sim = OrderBookSimulator.simulate_market_sell(book, sell_size)
            
            # Target is the price impact %
            target = sim["price_impact_pct"]
            
            feats = extract_features(book, sell_size)
            
            # Keep features and target
            # features list: [best_bid, spread, spread_pct, depth_bid_1..3, depth_ask_1..3, imbalance_1, imbalance_total, size_ratio1, size_ratio2, sell_size]
            data.append((feats, target))
            
    # Convert to numpy arrays
    X = np.array([item[0] for item in data], dtype=np.float32)
    y = np.array([item[1] for item in data], dtype=np.float32).reshape(-1, 1)
    
    # We have 14 features. Let's make sure PYTORCH_CONFIG input_dim matches
    return X, y

# --- ML Training Pipeline ---
class PriceImpactMLPipeline:
    def __init__(self):
        self.best_lgb_models = {}  # Store models for each quantile
        self.best_pytorch_model = None
        self.feature_names = [
            "best_bid", "spread", "spread_pct",
            "depth_bid_1", "depth_bid_2", "depth_bid_3",
            "depth_ask_1", "depth_ask_2", "depth_ask_3",
            "imbalance_1", "imbalance_total",
            "size_to_bid_1", "size_to_total_bid", "sell_size"
        ]

    def train_lightgbm_quantiles(self, X_train, y_train, X_val, y_val, quantiles=[0.1, 0.5, 0.9]):
        """
        Train LightGBM regressors with Quantile Loss for each target quantile.
        Performs grid search to find the best hyperparameters.
        """
        print("\n--- Training LightGBM Quantile Models ---")
        y_train_flat = y_train.flatten()
        y_val_flat = y_val.flatten()
        
        for q in quantiles:
            best_loss = float("inf")
            best_model = None
            best_params = {}
            
            # Grid Search Loop
            for n_est in LGBM_GRID["n_estimators"]:
                for lr in LGBM_GRID["learning_rate"]:
                    for num_l in LGBM_GRID["num_leaves"]:
                        for depth in LGBM_GRID["max_depth"]:
                            params = {
                                "objective": "quantile",
                                "alpha": q,
                                "metric": "quantile",
                                "n_estimators": n_est,
                                "learning_rate": lr,
                                "num_leaves": num_l,
                                "max_depth": depth,
                                "verbose": -1,
                                "random_state": 42
                            }
                            
                            model = lgb.LGBMRegressor(**params)
                            model.fit(X_train, y_train_flat)
                            
                            preds = model.predict(X_val)
                            loss = pinball_loss_np(y_val_flat, preds, q)
                            
                            if loss < best_loss:
                                best_loss = loss
                                best_model = model
                                best_params = params
                                
            print(f"Quantile {q:.1f}: Best Params = {best_params}, Validation Pinball Loss = {best_loss:.6f}")
            self.best_lgb_models[q] = best_model
            
        # Save models
        os.makedirs(os.path.dirname(LIGHTGBM_MODEL_PATH), exist_ok=True)
        joblib.dump(self.best_lgb_models, LIGHTGBM_MODEL_PATH)
        print(f"Saved LightGBM models to {LIGHTGBM_MODEL_PATH}")

    def train_pytorch_quantiles(self, X_train, y_train, X_val, y_val, quantiles=[0.1, 0.5, 0.9]):
        """
        Train a PyTorch Multi-Quantile MLP.
        """
        print("\n--- Training PyTorch Quantile Neural Network ---")
        input_dim = X_train.shape[1]
        
        # Grid search parameters to find best PyTorch config
        learning_rates = [0.001, 0.005, 0.01]
        hidden_options = [[64, 32], [128, 64, 32]]
        
        best_val_loss = float("inf")
        best_model_state = None
        best_lr = 0.005
        best_hidden = [64, 32]
        
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
        
        # Grid Search over PyTorch Hyperparameters
        for lr in learning_rates:
            for hidden in hidden_options:
                model = QuantileMLP(input_dim=input_dim, hidden_dims=hidden, output_dim=len(quantiles))
                criterion = QuantileLoss(quantiles=quantiles)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=PYTORCH_CONFIG.get("weight_decay", 1e-4))
                
                train_loader = DataLoader(train_dataset, batch_size=PYTORCH_CONFIG["batch_size"], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
                
                # Simple Train loop
                for epoch in range(50):  # limit epochs for grid search speed
                    model.train()
                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        preds = model(xb)
                        loss = criterion(preds, yb)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate on Val
                model.eval()
                with torch.no_grad():
                    val_x, val_y = next(iter(val_loader))
                    val_preds = model(val_x)
                    val_loss = criterion(val_preds, val_y).item()
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    best_lr = lr
                    best_hidden = hidden
                    
        print(f"PyTorch Grid Search finished. Best LR = {best_lr}, Best Hidden Layers = {best_hidden}, Val Loss = {best_val_loss:.6f}")
        
        # Train the best model fully
        self.best_pytorch_model = QuantileMLP(input_dim=input_dim, hidden_dims=best_hidden, output_dim=len(quantiles))
        self.best_pytorch_model.load_state_dict(best_model_state)
        
        # Final fine-tuning
        optimizer = optim.Adam(self.best_pytorch_model.parameters(), lr=best_lr, weight_decay=PYTORCH_CONFIG.get("weight_decay", 1e-4))
        criterion = QuantileLoss(quantiles=quantiles)
        train_loader = DataLoader(train_dataset, batch_size=PYTORCH_CONFIG["batch_size"], shuffle=True)
        
        for epoch in range(PYTORCH_CONFIG["epochs"]):
            self.best_pytorch_model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = self.best_pytorch_model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                
        # Save model
        torch.save({
            "model_state": self.best_pytorch_model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": best_hidden,
            "quantiles": quantiles
        }, PYTORCH_MODEL_PATH)
        print(f"Saved PyTorch model to {PYTORCH_MODEL_PATH}")

    def load_models(self):
        """Load pretrained LightGBM and PyTorch models."""
        if os.path.exists(LIGHTGBM_MODEL_PATH):
            self.best_lgb_models = joblib.load(LIGHTGBM_MODEL_PATH)
            
        if os.path.exists(PYTORCH_MODEL_PATH):
            checkpoint = torch.load(PYTORCH_MODEL_PATH)
            self.best_pytorch_model = QuantileMLP(
                input_dim=checkpoint["input_dim"],
                hidden_dims=checkpoint["hidden_dims"],
                output_dim=len(checkpoint["quantiles"])
            )
            self.best_pytorch_model.load_state_dict(checkpoint["model_state"])
            self.best_pytorch_model.eval()

    def predict_impact(self, order_book: dict, sell_size: float, model_type: str = "lightgbm"):
        """
        Predict price impact quantiles (10%, 50%, 90%) for a given order book and sell size.
        Returns a dictionary of predicted percentage drops.
        """
        feats = extract_features(order_book, sell_size).reshape(1, -1)
        
        if model_type == "lightgbm":
            if not self.best_lgb_models:
                self.load_models()
            if not self.best_lgb_models:
                # Fallback simple model
                return {0.1: sell_size * 0.0005, 0.5: sell_size * 0.001, 0.9: sell_size * 0.002}
                
            return {
                0.1: max(0.0, float(self.best_lgb_models[0.1].predict(feats)[0])),
                0.5: max(0.0, float(self.best_lgb_models[0.5].predict(feats)[0])),
                0.9: max(0.0, float(self.best_lgb_models[0.9].predict(feats)[0]))
            }
            
        elif model_type == "pytorch":
            if self.best_pytorch_model is None:
                self.load_models()
            if self.best_pytorch_model is None:
                return {0.1: sell_size * 0.0005, 0.5: sell_size * 0.001, 0.9: sell_size * 0.002}
                
            self.best_pytorch_model.eval()
            with torch.no_grad():
                tensor_feats = torch.tensor(feats)
                preds = self.best_pytorch_model(tensor_feats).numpy().flatten()
                
            return {
                0.1: max(0.0, float(preds[0])),
                0.5: max(0.0, float(preds[1])),
                0.9: max(0.0, float(preds[2]))
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

# --- Self-Test & Comparison Execution ---
if __name__ == "__main__":
    import random
    # Set seed
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    
    # Generate data
    X, y = generate_ml_dataset(n_snapshots=150)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    pipeline = PriceImpactMLPipeline()
    pipeline.train_lightgbm_quantiles(X_train, y_train, X_val, y_val)
    pipeline.train_pytorch_quantiles(X_train, y_train, X_val, y_val)
    
    # Simple verification evaluation
    print("\n--- Model Comparison on Test Data ---")
    lgb_preds = {q: pipeline.best_lgb_models[q].predict(X_val) for q in [0.1, 0.5, 0.9]}
    
    pipeline.best_pytorch_model.eval()
    with torch.no_grad():
        pytorch_preds = pipeline.best_pytorch_model(torch.tensor(X_val)).numpy()
        
    for i, q in enumerate([0.1, 0.5, 0.9]):
        lgb_loss = pinball_loss_np(y_val.flatten(), lgb_preds[q], q)
        pytorch_loss = pinball_loss_np(y_val.flatten(), pytorch_preds[:, i], q)
        print(f"Quantile {q:.1f} Pinball Loss: LightGBM = {lgb_loss:.6f} | PyTorch = {pytorch_loss:.6f}")
