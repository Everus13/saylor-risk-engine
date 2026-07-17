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
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=5y&interval=1d"
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

        # Target Label: 1 if Premium falls below 5% (0.05) within the next 30 days (excluding today, to avoid leakage), else 0
        # By shifting by -1, we look at days t+1 to t+30
        future_min_premium = df["premium"].shift(-1).iloc[::-1].rolling(window=30, min_periods=1).min().iloc[::-1]
        future_min_premium = future_min_premium.fillna(df["premium"])
        df["target"] = (future_min_premium < 0.05).astype(int)

        return df

    def augment_data(self, X: np.ndarray, y: np.ndarray, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples by resampling the original dataset with replacement
        and adding standard deviation scaled Gaussian noise (Jittering).
        """
        if len(X) == 0:
            return X, y
            
        indices = np.random.choice(len(X), size=n_samples, replace=True)
        aug_X = X[indices].copy()
        aug_y = y[indices].copy()
        
        # Calculate standard deviations of real features for scaling noise
        stds = np.std(X, axis=0)
        # Avoid zero division or scaling issues
        stds = np.where(stds == 0, 1e-4, stds)
        
        for col_idx in range(X.shape[1]):
            noise = np.random.normal(0, 0.05 * stds[col_idx], size=n_samples)
            aug_X[:, col_idx] += noise
            
        # Clip features to physical limits
        # Col mapping: 0=VIX, 1=DXY, 2=TNX, 3=premium_zscore, 4=macro_pressure_idx
        aug_X[:, 0] = np.clip(aug_X[:, 0], 5.0, 100.0)  # VIX
        aug_X[:, 1] = np.clip(aug_X[:, 1], 50.0, 150.0) # DXY
        aug_X[:, 2] = np.clip(aug_X[:, 2], 0.0, 15.0)   # TNX
        
        return aug_X, aug_y

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

        # Define features (excluding absolute premium SMAs to avoid trivial distance-to-default leakage)
        feature_cols = [
            "VIX", "DXY", "TNX", 
            "premium_zscore", "macro_pressure_idx"
        ]

        # Filter training to non-collapsed states (premium >= 5%) to predict onset of collapse
        filtered_df = df[df["premium"] >= 0.05]
        if len(filtered_df) < 50:
            logging.warning("Too few samples with premium >= 5%. Falling back to full dataset.")
            filtered_df = df

        X = filtered_df[feature_cols].fillna(0.0).values
        y = filtered_df["target"].values

        # Stratified K-Fold Cross-Validation (K=5) to prevent overfitting and resolve homogeneous test split
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        import math
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # 1. Evaluate Random Forest (with regularization: max_depth=5, min_samples_leaf=5 to prevent overfitting)
        rf_acc_scores, rf_f1_scores, rf_auc_scores = [], [], []
        for train_idx, val_idx in skf.split(X, y):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Augment training folds using synthetic generator to n_samples=5000
            X_train_cv_aug, y_train_cv_aug = self.augment_data(X_train_cv, y_train_cv, n_samples=5000)
            
            rf_cv = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
            rf_cv.fit(X_train_cv_aug, y_train_cv_aug)
            preds = rf_cv.predict(X_val_cv)
            
            rf_acc_scores.append(accuracy_score(y_val_cv, preds))
            rf_f1_scores.append(f1_score(y_val_cv, preds, zero_division=0))
            try:
                probs = rf_cv.predict_proba(X_val_cv)[:, 1]
                auc = roc_auc_score(y_val_cv, probs)
                if not math.isnan(auc):
                    rf_auc_scores.append(auc)
            except Exception:
                pass
                
        rf_acc = float(np.mean(rf_acc_scores)) if rf_acc_scores else 0.0
        rf_f1 = float(np.mean(rf_f1_scores)) if rf_f1_scores else 0.0
        rf_auc = float(np.mean(rf_auc_scores)) if rf_auc_scores else 0.5

        # 2. Evaluate LightGBM (with regularization: max_depth=4, min_child_samples=10, learning_rate=0.03)
        lgb_acc, lgb_f1, lgb_auc = 0.0, 0.0, 0.5
        lgb_trained = False
        try:
            from lightgbm import LGBMClassifier
            lgb_acc_scores, lgb_f1_scores, lgb_auc_scores = [], [], []
            for train_idx, val_idx in skf.split(X, y):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                # Augment training folds using synthetic generator to n_samples=5000
                X_train_cv_aug, y_train_cv_aug = self.augment_data(X_train_cv, y_train_cv, n_samples=5000)
                
                lgb_cv = LGBMClassifier(
                    n_estimators=100, learning_rate=0.03, max_depth=4, min_child_samples=10, verbose=-1, random_state=42
                )
                lgb_cv.fit(X_train_cv_aug, y_train_cv_aug)
                preds = lgb_cv.predict(X_val_cv)
                
                lgb_acc_scores.append(accuracy_score(y_val_cv, preds))
                lgb_f1_scores.append(f1_score(y_val_cv, preds, zero_division=0))
                try:
                    probs = lgb_cv.predict_proba(X_val_cv)[:, 1]
                    auc = roc_auc_score(y_val_cv, probs)
                    if not math.isnan(auc):
                        lgb_auc_scores.append(auc)
                except Exception:
                    pass
            lgb_acc = float(np.mean(lgb_acc_scores)) if lgb_acc_scores else 0.0
            lgb_f1 = float(np.mean(lgb_f1_scores)) if lgb_f1_scores else 0.0
            lgb_auc = float(np.mean(lgb_auc_scores)) if lgb_auc_scores else 0.5
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
            X_aug, y_aug = self.augment_data(X, y, n_samples=10000)
            if selected_model_type == "Random Forest":
                self.model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42)
            else:
                from lightgbm import LGBMClassifier
                self.model = LGBMClassifier(
                    n_estimators=100, learning_rate=0.03, max_depth=4, min_child_samples=10, verbose=-1, random_state=42
                )
                
            self.model.fit(X_aug, y_aug)
            
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
        if current_premium < 0.05:
            # Deterministic rule: if premium is already collapsed (< 5%), probability of collapse is 100%
            prob = 1.0
        elif self.model is not None:
            try:
                features_arr = np.array([[
                    vix, dxy, tnx, p_zscore, macro_pressure
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
