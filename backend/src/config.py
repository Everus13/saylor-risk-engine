import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Configuration Files
DEBT_SCHEDULE_PATH = os.path.join(CONFIG_DIR, "mstr_debt_schedule.json")

# Model Paths
LIGHTGBM_MODEL_PATH = os.path.join(DATA_DIR, "lgb_impact_model.bin")
PYTORCH_MODEL_PATH = os.path.join(DATA_DIR, "torch_impact_model.pth")
RL_MODEL_PATH = os.path.join(DATA_DIR, "rl_execution_model")
RL_MODEL_PATH_STANDARD = os.path.join(DATA_DIR, "ppo_standard")
RL_MODEL_PATH_STRESS = os.path.join(DATA_DIR, "ppo_stress")
RL_MODEL_PATH_LIVE = os.path.join(DATA_DIR, "ppo_live")
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, "btc_historical_2y.csv")
MNAV_MODEL_PATH = os.path.join(DATA_DIR, "mnav_rf_model.bin")
MNAV_HISTORY_PATH = os.path.join(DATA_DIR, "mnav_history_cache.csv")

# MicroStrategy Constants
MSTR_BTC_HOLDINGS_DEFAULT = 843775
MSTR_SHARES_OUTSTANDING_DEFAULT = 333913000

# Market Parameters
BTC_TICKER = "BTC-USD"
MSTR_TICKER = "MSTR"

# Machine Learning Hyperparameters
# Grid search grids for LightGBM Quantile Regression
LGBM_GRID = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 5, 10],
}

# PyTorch MLP Configuration
PYTORCH_CONFIG = {
    "input_dim": 13,           # OFI, spread, depth levels, size, etc.
    "hidden_dims": [64, 32],
    "lr": 0.005,
    "epochs": 100,
    "batch_size": 64,
    "weight_decay": 1e-4,
}

# RL parameters
RL_CONFIG = {
    "total_timesteps": 15000,
    "learning_rate": 1.5e-4,
    "batch_size": 128,
    "n_steps": 2048,
    "gamma": 0.99,
}
