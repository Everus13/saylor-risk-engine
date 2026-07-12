import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
import lightgbm as lgb
from src.ml_impact import generate_ml_dataset, extract_features, pinball_loss_np, QuantileLoss
from src.order_book_sim import OrderBookSimulator

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# --- 1. Architectural Definitions ---

# Architecture A: Baseline MLP (original)
class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Architecture B: Residual MLP (ResMLP)
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = self.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        return out

class ResMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU()
        )
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.output_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.output_layer(out)
        return out


# Architecture C: Deep MLP with Batch Normalization and GELU
class DeepBNGeluMLP(nn.Module):
    def __init__(self, input_dim, output_dim=3):
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
        # BatchNorm1d requires at least 2 samples. During validation if batch size is 1,
        # we check training mode or run in eval.
        return self.network(x)

# --- 2. Training Runner ---
def train_and_evaluate_model(model_class, X_train, y_train, X_val, y_val, name, epochs=120, lr=0.005):
    input_dim = X_train.shape[1]
    model = model_class(input_dim)
    criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_tensor_x = torch.tensor(X_val)
    val_tensor_y = torch.tensor(y_val)
    
    best_loss = float("inf")
    best_weights = None
    
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            
        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = model(val_tensor_x)
            val_loss = criterion(val_preds, val_tensor_y).item()
            
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.state_dict()
            
    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        final_val_preds = model(val_tensor_x).numpy()
        
    # Calculate pinball loss per quantile
    losses = {}
    for i, q in enumerate([0.1, 0.5, 0.9]):
        losses[q] = pinball_loss_np(y_val.flatten(), final_val_preds[:, i], q)
        
    print(f"[{name}] Val Pinball Loss: q_10={losses[0.1]:.6f}, q_50={losses[0.5]:.6f}, q_90={losses[0.9]:.6f} | Total={sum(losses.values()):.6f}")
    return losses, final_val_preds

# --- 3. LightGBM Baseline Runner ---
def train_lightgbm_baselines(X_train, y_train, X_val, y_val):
    y_train_flat = y_train.flatten()
    y_val_flat = y_val.flatten()
    
    lgb_preds = {}
    losses = {}
    
    for q in [0.1, 0.5, 0.9]:
        params = {
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "n_estimators": 100,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
            "random_state": 42
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train_flat)
        preds = model.predict(X_val)
        lgb_preds[q] = preds
        losses[q] = pinball_loss_np(y_val_flat, preds, q)
        
    print(f"[LightGBM Baseline] Val Pinball Loss: q_10={losses[0.1]:.6f}, q_50={losses[0.5]:.6f}, q_90={losses[0.9]:.6f} | Total={sum(losses.values()):.6f}")
    return losses, lgb_preds

# --- Main Experiments Script ---
if __name__ == "__main__":
    print("Generating training dataset (300 snapshots, ~1500 samples)...")
    X, y = generate_ml_dataset(n_snapshots=300)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"Dataset Split: Train={len(X_train)}, Val={len(X_val)}")
    
    # Run Baseline LightGBM
    lgb_losses, lgb_preds = train_lightgbm_baselines(X_train, y_train, X_val, y_val)
    print("-" * 60)
    
    # Run Model 1: Baseline MLP
    baseline_losses, _ = train_and_evaluate_model(
        BaselineMLP, X_train, y_train, X_val, y_val, "Baseline MLP"
    )
    
    # Run Model 2: ResMLP
    res_losses, _ = train_and_evaluate_model(
        ResMLP, X_train, y_train, X_val, y_val, "Residual MLP (ResMLP)"
    )
    
    # Run Model 3: Deep BN-GELU MLP
    deep_losses, _ = train_and_evaluate_model(
        DeepBNGeluMLP, X_train, y_train, X_val, y_val, "Deep BN-GELU MLP"
    )
    
    print("-" * 60)
    print("\n--- Summary Performance Comparison (Pinball Loss) ---")
    summary_df = pd.DataFrame({
        "Model": ["LightGBM (Prod)", "Baseline MLP", "Residual MLP (ResMLP)", "Deep BN-GELU MLP"],
        "Quantile 10%": [lgb_losses[0.1], baseline_losses[0.1], res_losses[0.1], deep_losses[0.1]],
        "Quantile 50%": [lgb_losses[0.5], baseline_losses[0.5], res_losses[0.5], deep_losses[0.5]],
        "Quantile 90%": [lgb_losses[0.9], baseline_losses[0.9], res_losses[0.9], deep_losses[0.9]],
        "Total loss": [
            sum(lgb_losses.values()), 
            sum(baseline_losses.values()), 
            sum(res_losses.values()), 
            sum(deep_losses.values())
        ]
    })
    print(summary_df.to_string(index=False))
