import json
import datetime
import logging
from typing import Dict, List, Any, Optional
import yfinance as yf
from src.config import DEBT_SCHEDULE_PATH, BTC_TICKER, MSTR_TICKER

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FinancialTracker:
    def __init__(self, schedule_path: str = DEBT_SCHEDULE_PATH) -> None:
        self.schedule_path: str = schedule_path
        self.data: Dict[str, Any] = self._load_schedule()
        
        # Override reserve from database if exists
        try:
            from backend.src.database import get_setting
            db_reserve = get_setting("usd_cash_reserve")
            if db_reserve:
                self.data["usd_cash_reserve"] = float(db_reserve)
        except ImportError:
            try:
                from src.database import get_setting
                db_reserve = get_setting("usd_cash_reserve")
                if db_reserve:
                    self.data["usd_cash_reserve"] = float(db_reserve)
            except Exception:
                pass
        except Exception:
            pass

    def _load_schedule(self) -> Dict[str, Any]:
        try:
            with open(self.schedule_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading debt schedule file: {e}")
            # Fallback mock data
            return {
                "usd_cash_reserve": 120000000.0,
                "quarterly_opex": 25000000.0,
                "preferred_stock": [],
                "convertible_notes": []
            }

    def update_cash_reserve(self, new_reserve: float) -> None:
        """Update USD cash reserve in memory and database."""
        self.data["usd_cash_reserve"] = new_reserve
        try:
            from backend.src.database import set_setting
            set_setting("usd_cash_reserve", str(new_reserve))
        except ImportError:
            try:
                from src.database import set_setting
                set_setting("usd_cash_reserve", str(new_reserve))
            except Exception:
                pass
        except Exception:
            pass

    def get_current_prices(self) -> Dict[str, float]:
        """Fetch current prices of BTC and MSTR using robust direct API query fallbacks."""
        prices = {"BTC": 60000.0, "MSTR": 150.0}
        import requests
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        # --- 1. Fetch Real-time BTC Price ---
        btc_fetched = False
        
        # Fallback A: Coinbase Spot Price API
        try:
            res = requests.get("https://api.coinbase.com/v2/prices/BTC-USD/spot", headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                prices["BTC"] = float(data["data"]["amount"])
                btc_fetched = True
                logging.info(f"Successfully fetched actual BTC price from Coinbase: ${prices['BTC']:.2f}")
        except Exception as e:
            logging.warning(f"Failed to fetch BTC price from Coinbase: {e}")

        # Fallback B: Blockchain.info Ticker API
        if not btc_fetched:
            try:
                res = requests.get("https://blockchain.info/ticker", headers=headers, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    prices["BTC"] = float(data["USD"]["last"])
                    btc_fetched = True
                    logging.info(f"Successfully fetched actual BTC price from Blockchain.info: ${prices['BTC']:.2f}")
            except Exception as e:
                logging.warning(f"Failed to fetch BTC price from Blockchain.info: {e}")

        # Fallback C: Binance CCXT
        if not btc_fetched:
            try:
                import ccxt
                exchange = ccxt.binance()
                ticker = exchange.fetch_ticker("BTC/USDT")
                if ticker and "last" in ticker and ticker["last"] is not None:
                    prices["BTC"] = float(ticker["last"])
                    btc_fetched = True
                    logging.info(f"Successfully fetched actual BTC price from Binance CCXT: ${prices['BTC']:.2f}")
            except Exception as e:
                logging.warning(f"Failed to fetch live BTC price via CCXT: {e}")

        # Fallback D: yfinance Ticker API
        if not btc_fetched:
            try:
                btc_ticker = yf.Ticker(BTC_TICKER)
                btc_info = btc_ticker.fast_info
                if btc_info and "last_price" in btc_info:
                    prices["BTC"] = float(btc_info["last_price"])
                else:
                    hist = btc_ticker.history(period="1d")
                    if not hist.empty:
                        prices["BTC"] = float(hist["Close"].iloc[-1])
            except Exception as e:
                logging.warning(f"Failed to fetch BTC price from yfinance: {e}. Using default 60000.0")

        # --- 2. Fetch Real-time MSTR Stock Price ---
        mstr_fetched = False
        
        # Fallback A: Direct Yahoo Finance Query Chart API (Bypasses yfinance block)
        try:
            res = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/MSTR?interval=1m&range=1d", headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                    mstr_price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
                    prices["MSTR"] = float(mstr_price)
                    mstr_fetched = True
                    logging.info(f"Successfully fetched actual MSTR price from Yahoo Chart API: ${prices['MSTR']:.2f}")
            else:
                logging.warning(f"Yahoo Chart API returned status code {res.status_code}")
        except Exception as e:
            logging.warning(f"Failed to fetch MSTR price directly from Yahoo Chart API: {e}")

        # Fallback B: Standard yfinance Ticker
        if not mstr_fetched:
            try:
                mstr_ticker = yf.Ticker(MSTR_TICKER)
                mstr_info = mstr_ticker.fast_info
                if mstr_info and "last_price" in mstr_info:
                    prices["MSTR"] = float(mstr_info["last_price"])
                else:
                    hist = mstr_ticker.history(period="1d")
                    if not hist.empty:
                        prices["MSTR"] = float(hist["Close"].iloc[-1])
            except Exception as e:
                logging.warning(f"Failed to fetch MSTR price from yfinance: {e}. Using default 150.0")

        return prices

    def calculate_obligations(self, start_date: datetime.date, days_forecast: int) -> Dict[str, Any]:
        """
        Calculate total cash required for MSTR obligations in the given forecast window.
        Returns a breakdown list of payments and total USD.
        """
        end_date = start_date + datetime.timedelta(days=days_forecast)
        payments: List[Dict[str, Any]] = []
        total_coupons = 0.0
        total_dividends = 0.0
        
        # Iterate day by day in the range to match scheduled payment dates
        curr_date = start_date
        while curr_date <= end_date:
            month = curr_date.month
            day = curr_date.day
            
            # Check Convertible Notes coupons
            for note in self.data.get("convertible_notes", []):
                maturity = int(note.get("maturity_year", 2030))
                if curr_date.year > maturity:
                    continue  # Already matured
                
                if month in note.get("payment_months", []) and day == note.get("payment_day", 15):
                    num_pmts = len(note.get("payment_months", [6, 12]))
                    coupon_amt = (float(note.get("coupon_rate", 0.0)) * float(note.get("principal", 0.0))) / num_pmts
                    if coupon_amt > 0:
                        payments.append({
                            "date": curr_date,
                            "type": "Купон конвертируемых облигаций",
                            "name": note.get("name"),
                            "amount_usd": coupon_amt
                        })
                        total_coupons += coupon_amt
            
            # Check Preferred Stock dividends
            for pref in self.data.get("preferred_stock", []):
                if month in pref.get("payment_months", []) and day == pref.get("payment_day", 15):
                    num_pmts = len(pref.get("payment_months", [3, 6, 9, 12]))
                    div_amt = (float(pref.get("annual_dividend_rate", 0.0)) * float(pref.get("liquidation_preference", 0.0))) / num_pmts
                    if div_amt > 0:
                        payments.append({
                            "date": curr_date,
                            "type": "Дивиденды по прив. акциям",
                            "name": pref.get("name"),
                            "amount_usd": div_amt
                        })
                        total_dividends += div_amt
                        
            curr_date += datetime.timedelta(days=1)

        total_usd = total_coupons + total_dividends

        return {
            "payments_breakdown": payments,
            "total_coupons_usd": total_coupons,
            "total_dividends_usd": total_dividends,
            "total_opex_usd": 0.0,
            "total_usd_required": total_usd
        }

    def get_btc_sell_requirements(self, days_forecast: int = 90, btc_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate total USD needed, subtract cash reserves, and convert the net to BTC to sell.
        """
        if btc_price is None:
            prices = self.get_current_prices()
            btc_price = prices["BTC"]

        start_date = datetime.date.today()
        result = self.calculate_obligations(start_date, days_forecast)
        
        total_usd = result["total_usd_required"]
        cash_reserve = float(self.data.get("usd_cash_reserve", 0.0))
        
        net_usd_needed = max(0.0, total_usd - cash_reserve)
        btc_to_sell = net_usd_needed / btc_price if btc_price > 0 else 0.0
        
        return {
            "days_forecast": days_forecast,
            "total_usd_required": total_usd,
            "usd_cash_reserve": cash_reserve,
            "net_usd_needed": net_usd_needed,
            "btc_price": btc_price,
            "btc_to_sell_stress_case": btc_to_sell,
            "btc_to_sell_normal_case": 0.0,
            "breakdown": result
        }

    def fetch_sec_edgar_facts(self) -> Dict[str, Any]:
        """
        Dynamically fetch MicroStrategy's latest reported financial facts from the SEC EDGAR API.
        """
        import requests
        url = "https://data.sec.gov/api/xbrl/companyfacts/CIK0001050446.json"
        headers = {
            "User-Agent": "MSTR-BTC Analytics Engine contact@mstr-btc-analytics.com"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                facts = data.get("facts", {}).get("us-gaap", {})
                
                parsed_facts: Dict[str, Any] = {}
                keys_mapping = {
                    "LongTermDebt": "long_term_debt",
                    "LongTermDebtCurrent": "long_term_debt_current",
                    "LongTermDebtNoncurrent": "long_term_debt_noncurrent",
                    "PreferredStockValue": "preferred_stock_value"
                }
                
                latest_date = None
                latest_form = None
                
                for xbrl_key, label in keys_mapping.items():
                    if xbrl_key in facts:
                        units = facts[xbrl_key].get("units", {})
                        for unit_name, reports in units.items():
                            if reports:
                                latest_rep = sorted(reports, key=lambda x: x.get('end', x.get('period', '')))[-1]
                                parsed_facts[label] = float(latest_rep.get("val", 0.0))
                                if latest_date is None or latest_rep.get('end', '') > latest_date:
                                    latest_date = latest_rep.get('end')
                                    latest_form = latest_rep.get('form')
                
                parsed_facts["date"] = latest_date
                parsed_facts["form"] = latest_form
                return parsed_facts
        except Exception as e:
            logging.error(f"Error fetching facts from SEC API: {e}")
            
        # Fallback values based on latest known Q1 2026 reports
        return {
            "long_term_debt": 8196524000.0,
            "long_term_debt_current": 31402000.0,
            "long_term_debt_noncurrent": 8165122000.0,
            "preferred_stock_value": 0.0,
            "date": "2026-03-31",
            "form": "10-Q (Кэшированные данные)"
        }

    def sync_obligations_with_sec(self, start_date: datetime.date, days_forecast: int, use_sec_debt: bool = False) -> Dict[str, Any]:
        """
        Calculate total cash required, optionally matching total debt outstanding to the SEC EDGAR facts.
        """
        end_date = start_date + datetime.timedelta(days=days_forecast)
        payments: List[Dict[str, Any]] = []
        total_coupons = 0.0
        total_dividends = 0.0
        
        notes = self.data.get("convertible_notes", [])
        
        # Calculate unallocated debt difference if sync enabled
        unallocated_principal = 0.0
        if use_sec_debt:
            sec_facts = self.fetch_sec_edgar_facts()
            sec_total_debt = float(sec_facts.get("long_term_debt", 0.0))
            
            # Sum of static notes active during this period
            static_total_principal = sum(float(note.get("principal", 0.0)) for note in notes if start_date.year <= int(note.get("maturity_year", 2030)))
            if sec_total_debt > static_total_principal:
                unallocated_principal = sec_total_debt - static_total_principal

        # Iterate day by day
        curr_date = start_date
        while curr_date <= end_date:
            month = curr_date.month
            day = curr_date.day
            
            # Check static Convertible Notes
            for note in notes:
                maturity = int(note.get("maturity_year", 2030))
                if curr_date.year > maturity:
                    continue
                
                if month in note.get("payment_months", []) and day == note.get("payment_day", 15):
                    num_pmts = len(note.get("payment_months", [6, 12]))
                    coupon_amt = (float(note.get("coupon_rate", 0.0)) * float(note.get("principal", 0.0))) / num_pmts
                    if coupon_amt > 0:
                        payments.append({
                            "date": curr_date,
                            "type": "Купон конвертируемых облигаций",
                            "name": note.get("name"),
                            "amount_usd": coupon_amt
                        })
                        total_coupons += coupon_amt
            
            # Add coupon payments for unallocated SEC debt (assuming average coupon rate of 1.0% paid semi-annually in June/December)
            if unallocated_principal > 0:
                if month in [6, 12] and day == 15:
                    unallocated_coupon = (0.01 * unallocated_principal) / 2.0
                    payments.append({
                        "date": curr_date,
                        "type": "Купон неучтенного долга по SEC",
                        "name": "Нераспределенная долговая разница из SEC баланса",
                        "amount_usd": unallocated_coupon
                    })
                    total_coupons += unallocated_coupon
            
            # Check Preferred Stock dividends
            for pref in self.data.get("preferred_stock", []):
                if month in pref.get("payment_months", []) and day == pref.get("payment_day", 15):
                    num_pmts = len(pref.get("payment_months", [3, 6, 9, 12]))
                    div_amt = (float(pref.get("annual_dividend_rate", 0.0)) * float(pref.get("liquidation_preference", 0.0))) / num_pmts
                    if div_amt > 0:
                        payments.append({
                            "date": curr_date,
                            "type": "Дивиденды по прив. акциям",
                            "name": pref.get("name"),
                            "amount_usd": div_amt
                        })
                        total_dividends += div_amt
                        
            curr_date += datetime.timedelta(days=1)

        total_usd = total_coupons + total_dividends

        return {
            "payments_breakdown": payments,
            "total_coupons_usd": total_coupons,
            "total_dividends_usd": total_dividends,
            "total_opex_usd": 0.0,
            "total_usd_required": total_usd
        }

    def get_btc_sell_requirements_sync(self, days_forecast: int = 90, btc_price: Optional[float] = None, use_sec_debt: bool = False) -> Dict[str, Any]:
        """
        Calculate target BTC to sell, dynamically syncing with SEC facts if specified.
        """
        if btc_price is None:
            prices = self.get_current_prices()
            btc_price = prices["BTC"]

        start_date = datetime.date.today()
        result = self.sync_obligations_with_sec(start_date, days_forecast, use_sec_debt=use_sec_debt)
        
        total_usd = result["total_usd_required"]
        cash_reserve = float(self.data.get("usd_cash_reserve", 0.0))
        
        net_usd_needed = max(0.0, total_usd - cash_reserve)
        btc_to_sell = net_usd_needed / btc_price if btc_price > 0 else 0.0
        
        return {
            "days_forecast": days_forecast,
            "total_usd_required": total_usd,
            "usd_cash_reserve": cash_reserve,
            "net_usd_needed": net_usd_needed,
            "btc_price": btc_price,
            "btc_to_sell_stress_case": btc_to_sell,
            "btc_to_sell_normal_case": 0.0,
            "breakdown": result
        }
