import datetime
import json
import os
import logging
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Ensure DB folder exists
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "mstr_btc.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---

class Setting(Base):
    __tablename__ = "settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    value = Column(String, nullable=False)

class SimulationRecord(Base):
    __tablename__ = "simulation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    strategy = Column(String, nullable=False)  # "rl", "twap", "l2_walk"
    sell_volume = Column(Float, nullable=False)
    steps = Column(Integer, nullable=False)
    avg_price = Column(Float, nullable=False)
    total_revenue = Column(Float, nullable=False)
    slippage_pct = Column(Float, nullable=False)
    details_json = Column(Text, nullable=False)  # JSON-encoded array of steps

# Create tables
Base.metadata.create_all(bind=engine)

# --- Helper DB Functions ---

def get_setting(key: str, default: str = "") -> str:
    db = SessionLocal()
    try:
        db_item = db.query(Setting).filter(Setting.key == key).first()
        if db_item:
            return db_item.value
        return default
    except Exception as e:
        logging.error(f"DB Error getting setting {key}: {e}")
        return default
    finally:
        db.close()

def set_setting(key: str, value: str) -> None:
    db = SessionLocal()
    try:
        db_item = db.query(Setting).filter(Setting.key == key).first()
        if db_item:
            db_item.value = value
        else:
            db_item = Setting(key=key, value=value)
            db.add(db_item)
        db.commit()
    except Exception as e:
        logging.error(f"DB Error setting {key}={value}: {e}")
        db.rollback()
    finally:
        db.close()

def add_simulation_record(
    strategy: str,
    sell_volume: float,
    steps: int,
    avg_price: float,
    total_revenue: float,
    slippage_pct: float,
    details: List[Dict[str, Any]]
) -> None:
    db = SessionLocal()
    try:
        record = SimulationRecord(
            strategy=strategy,
            sell_volume=sell_volume,
            steps=steps,
            avg_price=avg_price,
            total_revenue=total_revenue,
            slippage_pct=slippage_pct,
            details_json=json.dumps(details)
        )
        db.add(record)
        db.commit()
    except Exception as e:
        logging.error(f"DB Error adding simulation record: {e}")
        db.rollback()
    finally:
        db.close()

def get_simulation_history(limit: int = 20) -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        records = db.query(SimulationRecord).order_by(SimulationRecord.timestamp.desc()).limit(limit).all()
        result = []
        for r in records:
            result.append({
                "id": r.id,
                "timestamp": r.timestamp.isoformat(),
                "strategy": r.strategy,
                "sell_volume": r.sell_volume,
                "steps": r.steps,
                "avg_price": r.avg_price,
                "total_revenue": r.total_revenue,
                "slippage_pct": r.slippage_pct,
                "details": json.loads(r.details_json)
            })
        return result
    except Exception as e:
        logging.error(f"DB Error getting simulation history: {e}")
        return []
    finally:
        db.close()
