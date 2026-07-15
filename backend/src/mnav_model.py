import os
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

# Constants
from src.config import DATA_DIR

MSTR_BTC_HOLDINGS_DEFAULT = 843775
MSTR_SHARES_OUTSTANDING_DEFAULT = 333913000

class MNAVModelPipeline:
    def __init__(self, data_dir: str = DATA_DIR) -> None:
        self.data_dir = data_dir
        self.model_path = os.path.join(data_dir, "mnav_rf_model.bin")
        self.history_path = os.path.join(data_dir, "mnav_history_cache.csv")
        self.model = None

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch 2 years of daily close prices for MSTR, BTC-USD, VIX, DXY, and TNX.
        Uses direct API requests to bypass potential standard scraping blocks.
        """
        import requests
        tickers = {
            "BTC": "BTC-USD",
            "MSTR": "MSTR",
            "VIX": "%5EVIX",       # ^VIX encoded
            "DXY": "DX-Y.NYB",
            "TNX": "%5ETNX"        # ^TNX encoded
        }
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        all_series = {}

        for key, ticker in tickers.items():
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
            try:
                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    result = data["chart"]["result"][0]
                    timestamps = result["timestamp"]
                    closes = result["indicators"]["quote"][0]["close"]
                    
                    dates = pd.to_datetime(timestamps, unit='s').date
                    series = pd.Series(closes, index=dates, name=key)
                    all_series[key] = series.dropna()
            except Exception as e:
                logging.error(f"Error downloading {ticker} history: {e}")

        if not all_series:
            # Return empty DataFrame if all downloads fail
            return pd.DataFrame()

        # Join all series on BTC dates (24/7) and forward-fill weekend stock closes
        df = pd.DataFrame(index=all_series.get("BTC", pd.Series(dtype=float)).index)
        for key, series in all_series.items():
            df = df.join(series, how="left")
        
        df = df.ffill().bfill()
        return df

    def calculate_mnav_features(
        self, 
        df: pd.DataFrame, 
        btc_holdings: float, 
        shares_outstanding: float, 
        cash_reserve: float, 
        long_term_debt: float
    ) -> pd.DataFrame:
        """
        Calculate daily mNAV and Premium values, and engineer feature indicators.
        """
        df = df.copy()
        
        # Calculate per-share metrics
        implied_btc_per_share = btc_holdings / shares_outstanding
        cash_per_share = cash_reserve / shares_outstanding
        debt_per_share = long_term_debt / shares_outstanding

        # Calculate mNAV and Premium
        df["mnav"] = (implied_btc_per_share * df["BTC"]) + cash_per_share - debt_per_share
        df["premium"] = (df["MSTR"] / df["mnav"]) - 1.0

        # Feature Engineering
        df["premium_sma_7"] = df["premium"].rolling(window=7, min_periods=1).mean()
        df["premium_sma_14"] = df["premium"].rolling(window=14, min_periods=1).mean()
        df["premium_sma_30"] = df["premium"].rolling(window=30, min_periods=1).mean()
        
        df["premium_mean_90"] = df["premium"].rolling(window=90, min_periods=1).mean()
        df["premium_std_90"] = df["premium"].rolling(window=90, min_periods=1).std().replace(0, 1e-4).fillna(1e-4)
        df["premium_zscore"] = (df["premium"] - df["premium_mean_90"]) / df["premium_std_90"]
        
        # Macro indicators
        df["macro_pressure_idx"] = (df["VIX"] / 20.0) + (df["TNX"] / 4.0)

        # Target Label: 1 if Premium falls below 5% (0.05) within the next 30 days, else 0
        future_min_premium = df["premium"].iloc[::-1].rolling(window=30, min_periods=1).min().iloc[::-1]
        df["target"] = (future_min_premium < 0.05).astype(int)

        return df

    def train_pipeline(
        self,
        btc_holdings: float = MSTR_BTC_HOLDINGS_DEFAULT,
        shares_outstanding: float = MSTR_SHARES_OUTSTANDING_DEFAULT,
        cash_reserve: float = 3000000000.0,
        long_term_debt: float = 8196524000.0
    ) -> Dict[str, Any]:
        """
        Download history, extract features, compare Random Forest vs LightGBM on split data,
        select the optimal model based on F1-score, train on full data, and save.
        """
        df = self.fetch_data()
        if df.empty or len(df) < 100:
            logging.error("Insufficient history data to train mNAV pipeline.")
            return {"status": "error", "detail": "Insufficient history data to train mNAV pipeline."}

        df = self.calculate_mnav_features(
            df, btc_holdings, shares_outstanding, cash_reserve, long_term_debt
        )
        df.to_csv(self.history_path)

        # Define features
        feature_cols = [
            "VIX", "DXY", "TNX", 
            "premium_sma_7", "premium_sma_14", "premium_sma_30", "premium_zscore", "macro_pressure_idx"
        ]

        X = df[feature_cols].fillna(0.0).values
        y = df["target"].values

        # 80/20 Chronological Split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        # 1. Evaluate Random Forest
        rf_test = RandomForestClassifier(n_estimators=150, random_state=42)
        rf_test.fit(X_train, y_train)
        rf_preds = rf_test.predict(X_test)
        
        import math
        try:
            rf_probs = rf_test.predict_proba(X_test)[:, 1]
            rf_auc = float(roc_auc_score(y_test, rf_probs))
            if math.isnan(rf_auc):
                rf_auc = 0.5
        except Exception:
            rf_auc = 0.5
            
        rf_acc = float(accuracy_score(y_test, rf_preds))
        rf_f1 = float(f1_score(y_test, rf_preds, zero_division=0))

        # 2. Evaluate LightGBM
        lgb_acc, lgb_f1, lgb_auc = 0.0, 0.0, 0.5
        lgb_trained = False
        try:
            from lightgbm import LGBMClassifier
            lgb_test = LGBMClassifier(n_estimators=150, learning_rate=0.05, verbose=-1, random_state=42)
            lgb_test.fit(X_train, y_train)
            lgb_preds = lgb_test.predict(X_test)
            try:
                lgb_probs = lgb_test.predict_proba(X_test)[:, 1]
                lgb_auc = float(roc_auc_score(y_test, lgb_probs))
                if math.isnan(lgb_auc):
                    lgb_auc = 0.5
            except Exception:
                lgb_auc = 0.5
            lgb_acc = float(accuracy_score(y_test, lgb_preds))
            lgb_f1 = float(f1_score(y_test, lgb_preds, zero_division=0))
            lgb_trained = True
        except Exception as e:
            logging.warning(f"LightGBM evaluation failed: {e}. RF will be used exclusively.")

        # Compare and select
        selected_model_type = "Random Forest"
        comparison_summary = f"RF (F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}) vs LGBM (F1: {lgb_f1:.4f}, AUC: {lgb_auc:.4f})"
        
        if lgb_trained and lgb_f1 > rf_f1:
            selected_model_type = "LightGBM"
            
        logging.info(f"Model Selection Result: {comparison_summary}. Chosen: {selected_model_type}")

        # Retrain selected model on full dataset
        try:
            if selected_model_type == "Random Forest":
                self.model = RandomForestClassifier(n_estimators=150, random_state=42)
            else:
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(n_estimators=150, learning_rate=0.05, verbose=-1, random_state=42)
                
            self.model.fit(X, y)
            
            # Save model with joblib
            import joblib
            os.makedirs(self.data_dir, exist_ok=True)
            joblib.dump((self.model, feature_cols, selected_model_type, {
                "rf_accuracy": rf_acc, "rf_f1": rf_f1, "rf_auc": rf_auc,
                "lgb_accuracy": lgb_acc, "lgb_f1": lgb_f1, "lgb_auc": lgb_auc,
                "chosen": selected_model_type
            }), self.model_path)
            
            logging.info(f"Successfully trained full {selected_model_type} model.")
            return {
                "status": "success",
                "chosen": selected_model_type,
                "rf": {"accuracy": rf_acc, "f1_score": rf_f1, "roc_auc": rf_auc},
                "lgb": {"accuracy": lgb_acc, "f1_score": lgb_f1, "roc_auc": lgb_auc}
            }
        except Exception as e:
            logging.error(f"Error saving optimized model: {e}")
            return {"status": "error", "detail": str(e)}

    def load_model(self) -> bool:
        if self.model is not None:
            return True
        if os.path.exists(self.model_path):
            try:
                import joblib
                loaded = joblib.load(self.model_path)
                if isinstance(loaded, tuple) and len(loaded) == 4:
                    self.model, _, self.model_name, self.metrics = loaded
                    # Clean up any potential NaN values from metrics dictionary for JSON compliance
                    import math
                    if isinstance(self.metrics, dict):
                        for k, v in list(self.metrics.items()):
                            if isinstance(v, float) and math.isnan(v):
                                self.metrics[k] = 0.5
                else:
                    self.model = loaded[0] if isinstance(loaded, tuple) else loaded
                    self.model_name = "Random Forest"
                    self.metrics = {}
                return True
            except Exception as e:
                logging.error(f"Error loading mNAV model: {e}")
        return False

    def predict_collapse_probability(
        self,
        current_btc: float,
        current_mstr: float,
        current_vix: float,
        current_dxy: float,
        current_tnx: float,
        btc_holdings: float,
        shares_outstanding: float,
        cash_reserve: float,
        long_term_debt: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Returns:
          - probability: Chance of Premium Collapse (< 5%) in the next 30 days (0.0 to 1.0).
          - features: Dict of engineered feature values.
        """
        self.load_model()
        
        # Load historical context to calculate SMA / Z-score
        history_df = None
        if os.path.exists(self.history_path):
            try:
                history_df = pd.read_csv(self.history_path, index_col=0)
            except Exception:
                pass
                
        if history_df is None or history_df.empty:
            # Fallback training if no history exists
            self.train_pipeline(btc_holdings, shares_outstanding, cash_reserve, long_term_debt)
            try:
                history_df = pd.read_csv(self.history_path, index_col=0)
            except Exception:
                history_df = pd.DataFrame()

        # Calculate current metrics
        implied_btc_per_share = btc_holdings / shares_outstanding
        cash_per_share = cash_reserve / shares_outstanding
        debt_per_share = long_term_debt / shares_outstanding
        current_mnav = (implied_btc_per_share * current_btc) + cash_per_share - debt_per_share
        current_premium = (current_mstr / current_mnav) - 1.0

        # Feature values
        vix = current_vix
        dxy = current_dxy
        tnx = current_tnx
        
        # Approximate SMA/Zscore with historical context
        if not history_df.empty and "premium" in history_df.columns:
            recent_prems = history_df["premium"].tail(90).tolist()
            recent_prems.append(current_premium)
            
            p_sma7 = float(np.mean(recent_prems[-7:]))
            p_sma14 = float(np.mean(recent_prems[-14:]))
            p_sma30 = float(np.mean(recent_prems[-30:]))
            
            p_mean90 = float(np.mean(recent_prems))
            p_std90 = float(np.std(recent_prems)) if np.std(recent_prems) > 0 else 1e-4
            p_zscore = (current_premium - p_mean90) / p_std90
        else:
            p_sma7 = current_premium
            p_sma14 = current_premium
            p_sma30 = current_premium
            p_zscore = 0.0

        macro_pressure = (vix / 20.0) + (tnx / 4.0)

        feature_values = {
            "VIX": vix,
            "DXY": dxy,
            "TNX": tnx,
            "premium_sma_7": p_sma7,
            "premium_sma_14": p_sma14,
            "premium_sma_30": p_sma30,
            "premium_zscore": p_zscore,
            "macro_pressure_idx": macro_pressure
        }

        # Predict probability
        prob = 0.0
        if self.model is not None:
            try:
                features_arr = np.array([[
                    vix, dxy, tnx, p_sma7, p_sma14, p_sma30, p_zscore, macro_pressure
                ]])
                prob = float(self.model.predict_proba(features_arr)[0][1])
            except Exception as e:
                logging.error(f"Prediction failed, falling back: {e}")
                # Simple rule fallback
                if current_premium < 0.08:
                    prob = 0.85
                elif current_premium < 0.15:
                    prob = 0.40
                else:
                    prob = 0.10
        else:
            # Fallback if no model is trained
            if current_premium < 0.08:
                prob = 0.85
            elif current_premium < 0.15:
                prob = 0.40
            else:
                prob = 0.10

        return prob, {
            "implied_btc_per_share": implied_btc_per_share,
            "mnav_per_share": current_mnav,
            "premium_pct": current_premium * 100.0,
            "vix": vix,
            "dxy": dxy,
            "tnx": tnx,
            "macro_pressure_idx": macro_pressure
        }
