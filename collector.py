"""
collector.py — Data collector for aggregated CVD scanner.

Fetches OHLC + taker buy volume from multiple exchanges and stores in Parquet.
Runs continuously, updating every 60 seconds.

Output files:
  data/btc_spot_5m.parquet     columns: timestamp, exchange, open, high, low, close, volume, taker_buy_vol
  data/btc_futures_5m.parquet  columns: timestamp, exchange, open, high, low, close, volume, taker_buy_vol
  data/btc_oi_1m.parquet       columns: timestamp, exchange, oi_value

Spot exchanges:
  Native taker buy (klines): binance
  WebSocket trade stream:    bybit_spot, okx_spot, gate_spot, mexc_spot, coinbase, kraken

Futures exchanges:
  Native taker buy (klines): binance_futures
  WebSocket trade stream:    bybit_futures, okx_futures, gate_futures, mexc_futures

OI: binance_futures (polled every 60s, backfilled via 5m historical endpoint)

WebSocket design:
  - One persistent WsRunner thread per exchange (daemon)
  - Automatic reconnect with exponential backoff (5s → 60s max)
  - Application-level heartbeat for exchanges that require it (Bybit, OKX, MEXC futures)
  - WebSocket-level ping frames (ping_interval=30s) for all connections
  - Health monitor: forces reconnect if no trades received for WS_STALE_S seconds
  - REST fetchers kept for initial warmup snapshot on startup
"""

import json
import os
import threading
import time
import logging

import requests
import pandas as pd
import websocket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional
from dotenv import load_dotenv

load_dotenv()
_TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
_TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_DIR        = Path("data")
SIGNAL_LOG_FILE = DATA_DIR / "signal_log.csv"
HISTORY_DAYS = 5
INTERVAL_5M_MS = 5 * 60 * 1000
CANDLE_INTERVAL = "5min"        # pandas resample string
UPDATE_INTERVAL_S = 60          # seconds between collection cycles
TRADES_BUFFER_MINUTES = 20      # how long to keep raw trades in memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def now_ms() -> int:
    """Current UTC time in milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def ms_to_dt(ms: int) -> datetime:
    """Milliseconds to UTC datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def dt_to_ms(dt: datetime) -> int:
    """UTC datetime to milliseconds."""
    return int(dt.timestamp() * 1000)


def history_start_ms() -> int:
    """Timestamp for HISTORY_DAYS ago."""
    return dt_to_ms(datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS))


def load_parquet(filepath: Path) -> pd.DataFrame:
    """Load parquet file, return empty DataFrame if not found."""
    if filepath.exists():
        return pd.read_parquet(filepath)
    return pd.DataFrame()


def save_parquet(df: pd.DataFrame, filepath: Path) -> None:
    """Save DataFrame to parquet using atomic write to prevent partial reads."""
    if df.empty:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp.parquet")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, filepath)


def upsert_parquet(filepath: Path, new_df: pd.DataFrame) -> int:
    """
    Append new_df to existing parquet file.
    Deduplicates by (timestamp, exchange) and keeps the latest value.
    Returns number of rows in new_df.
    """
    if new_df.empty:
        return 0
    existing = load_parquet(filepath)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp", "exchange"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    save_parquet(combined, filepath)
    return len(new_df)


def last_stored_ts(filepath: Path, exchange: str) -> int:
    """
    Return the last stored timestamp (ms) for a given exchange in a parquet file.
    Falls back to history_start_ms() if no data found.
    """
    df = load_parquet(filepath)
    if df.empty:
        return history_start_ms()
    subset = df[df["exchange"] == exchange]
    if subset.empty:
        return history_start_ms()
    return int(subset["timestamp"].max().timestamp() * 1000)


# ---------------------------------------------------------------------------
# SPOT — NATIVE TAKER BUY (BINANCE)
# Uses Binance-compatible klines endpoint: field [9] = taker_buy_vol
# ---------------------------------------------------------------------------

SPOT_NATIVE_ENDPOINTS = {
    "binance": "https://api.binance.com/api/v3/klines",
}


def fetch_spot_klines_native(exchange: str, start_ms: int) -> pd.DataFrame:
    """Fetch spot klines with native taker buy volume (Binance)."""
    url = SPOT_NATIVE_ENDPOINTS[exchange]
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": 1000, "startTime": start_ms}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_ts", "quote_vol", "n_trades", "taker_buy_vol", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
    df["exchange"] = exchange
    for col in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "exchange", "open", "high", "low", "close", "volume", "taker_buy_vol"]]


def backfill_spot_native(exchange: str) -> None:
    """Backfill spot klines. On first run: 5 days. On restart: only missing candles."""
    filepath = DATA_DIR / "btc_spot_5m.parquet"
    start_ms = last_stored_ts(filepath, exchange)
    all_dfs = []

    log.info(f"[{exchange}] Backfilling spot from {ms_to_dt(start_ms).strftime('%Y-%m-%d %H:%M')}...")
    while start_ms < now_ms():
        df = fetch_spot_klines_native(exchange, start_ms)
        if df.empty:
            break
        all_dfs.append(df)
        start_ms = int(df["timestamp"].max().timestamp() * 1000) + INTERVAL_5M_MS
        if len(df) < 1000:
            break
        time.sleep(0.2)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        upsert_parquet(filepath, result)
        log.info(f"[{exchange}] Spot backfill done: {len(result)} candles")


def update_spot_native(exchange: str) -> None:
    """Fetch only new spot klines since last stored timestamp."""
    filepath = DATA_DIR / "btc_spot_5m.parquet"
    start_ms = last_stored_ts(filepath, exchange)
    df = fetch_spot_klines_native(exchange, start_ms)
    if not df.empty:
        n = upsert_parquet(filepath, df)
        log.info(f"[{exchange}] spot: +{n} candles")


# ---------------------------------------------------------------------------
# FUTURES — NATIVE TAKER BUY (BINANCE FUTURES)
# ---------------------------------------------------------------------------

def fetch_futures_klines_binance(start_ms: int) -> pd.DataFrame:
    """Fetch Binance Futures klines with native taker buy volume."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": "BTCUSDT", "interval": "5m", "limit": 1500, "startTime": start_ms}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=[
        "ts", "open", "high", "low", "close", "volume",
        "close_ts", "quote_vol", "n_trades", "taker_buy_vol", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
    df["exchange"] = "binance_futures"
    for col in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "exchange", "open", "high", "low", "close", "volume", "taker_buy_vol"]]


def backfill_futures_binance() -> None:
    """Backfill Binance Futures klines. On first run: 5 days. On restart: only missing candles."""
    filepath = DATA_DIR / "btc_futures_5m.parquet"
    start_ms = last_stored_ts(filepath, "binance_futures")
    all_dfs = []

    log.info(f"[binance_futures] Backfilling futures from {ms_to_dt(start_ms).strftime('%Y-%m-%d %H:%M')}...")
    while start_ms < now_ms():
        df = fetch_futures_klines_binance(start_ms)
        if df.empty:
            break
        all_dfs.append(df)
        start_ms = int(df["timestamp"].max().timestamp() * 1000) + INTERVAL_5M_MS
        if len(df) < 1500:
            break
        time.sleep(0.2)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        upsert_parquet(filepath, result)
        log.info(f"[binance_futures] Futures backfill done: {len(result)} candles")


def update_futures_binance() -> None:
    """Fetch only new Binance Futures klines since last stored timestamp."""
    filepath = DATA_DIR / "btc_futures_5m.parquet"
    start_ms = last_stored_ts(filepath, "binance_futures")
    df = fetch_futures_klines_binance(start_ms)
    if not df.empty:
        n = upsert_parquet(filepath, df)
        log.info(f"[binance_futures] +{n} candles")


# ---------------------------------------------------------------------------
# REST TRADE FETCHERS — used only for initial warmup snapshot on startup
# These seed the buffer before WebSocket connections come online.
# ---------------------------------------------------------------------------

def fetch_trades_mexc_spot() -> pd.DataFrame:
    """
    Fetch recent BTC/USDT spot trades from MEXC.
    MEXC klines return only 8 columns (no taker buy split), so trades are used instead.
    isBuyerMaker=True means the buyer was the maker → the taker was a seller → NOT a taker buy.
    """
    url = "https://api.mexc.com/api/v3/trades"
    params = {"symbol": "BTCUSDT", "limit": 1000}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    trades = resp.json()
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["time"].astype(float), unit="ms", utc=True)
    df["price"] = df["price"].astype(float)
    df["size"] = df["qty"].astype(float)
    df["is_buyer"] = ~df["isBuyerMaker"].astype(bool)  # taker buy = maker was NOT the buyer
    return df[["timestamp", "price", "size", "is_buyer"]]


def fetch_trades_bybit(category: str = "spot", symbol: str = "BTCUSDT") -> pd.DataFrame:
    """
    Fetch recent trades from Bybit.
    category: 'spot' or 'linear' (perpetual futures)
    """
    url = "https://api.bybit.com/v5/market/recent-trade"
    params = {"category": category, "symbol": symbol, "limit": 1000}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("retCode") != 0:
        log.warning(f"Bybit trades error: {data.get('retMsg')}")
        return pd.DataFrame()

    trades = data["result"]["list"]
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["time"].astype(float), unit="ms", utc=True)
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df["is_buyer"] = df["side"] == "Buy"
    return df[["timestamp", "price", "size", "is_buyer"]]


def fetch_trades_okx(inst_id: str = "BTC-USDT") -> pd.DataFrame:
    """
    Fetch recent trades from OKX.
    inst_id: 'BTC-USDT' for spot, 'BTC-USDT-SWAP' for perpetual futures.

    Units:
      Spot (BTC-USDT):      sz is in BTC — no conversion needed
      Futures (SWAP):       sz is in contracts, 1 contract = 0.01 BTC — multiply by 0.01
    """
    url = "https://www.okx.com/api/v5/market/trades"
    params = {"instId": inst_id, "limit": 500}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "0":
        log.warning(f"OKX trades error: {data.get('msg')}")
        return pd.DataFrame()

    trades = data["data"]
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["ts"].astype(float), unit="ms", utc=True)
    df["price"] = df["px"].astype(float)
    df["size"] = df["sz"].astype(float)

    # Futures contracts → BTC
    if "SWAP" in inst_id:
        df["size"] = df["size"] * 0.01

    df["is_buyer"] = df["side"] == "buy"
    return df[["timestamp", "price", "size", "is_buyer"]]


def fetch_trades_gate_spot() -> pd.DataFrame:
    """Fetch recent spot trades from Gate.io."""
    url = "https://api.gateio.ws/api/v4/spot/trades"
    params = {"currency_pair": "BTC_USDT", "limit": 1000}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    trades = resp.json()
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["create_time_ms"].astype(float), unit="ms", utc=True)
    df["price"] = df["price"].astype(float)
    df["size"] = df["amount"].astype(float)
    df["is_buyer"] = df["side"] == "buy"
    return df[["timestamp", "price", "size", "is_buyer"]]


def fetch_trades_gate_futures() -> pd.DataFrame:
    """Fetch recent USDT perpetual futures trades from Gate.io."""
    url = "https://api.gateio.ws/api/v4/futures/usdt/trades"
    params = {"contract": "BTC_USDT", "limit": 1000}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    trades = resp.json()
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    # Gate API returns create_time_ms as float seconds (not ms despite the name) — multiply by 1000
    df["timestamp"] = pd.to_datetime(df["create_time"].astype(float) * 1000, unit="ms", utc=True)
    df["price"] = df["price"].astype(float)
    df["size_raw"] = df["size"].astype(float)
    # Gate futures: positive size = buy, negative size = sell (no separate 'side' field)
    df["is_buyer"] = df["size_raw"] > 0
    # Gate futures: size is in contracts, 1 contract = 1 USD → divide by price to get BTC
    df["size"] = df["size_raw"].abs() / df["price"]
    return df[["timestamp", "price", "size", "is_buyer"]]


def fetch_trades_coinbase() -> pd.DataFrame:
    """Fetch recent BTC-USD trades from Coinbase Exchange (public API)."""
    url = "https://api.exchange.coinbase.com/products/BTC-USD/trades"
    params = {"limit": 1000}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    trades = resp.json()
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True)
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    # Coinbase side: "buy" means the aggressor was a buyer (taker buy)
    df["is_buyer"] = df["side"] == "buy"
    return df[["timestamp", "price", "size", "is_buyer"]]


def fetch_trades_kraken() -> pd.DataFrame:
    """Fetch recent BTC/USD trades from Kraken (last ~1000 trades)."""
    url = "https://api.kraken.com/0/public/Trades"
    params = {"pair": "XBTUSD"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        log.warning(f"Kraken trades error: {data['error']}")
        return pd.DataFrame()

    result = data["result"]
    pair_key = next((k for k in result if k != "last"), None)
    if not pair_key:
        return pd.DataFrame()

    trades = result[pair_key]
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades, columns=["price", "volume", "time", "side", "order_type", "misc", "trade_id"])
    df["timestamp"] = pd.to_datetime(df["time"].astype(float), unit="s", utc=True)
    df["price"] = df["price"].astype(float)
    df["size"] = df["volume"].astype(float)
    df["is_buyer"] = df["side"] == "b"
    return df[["timestamp", "price", "size", "is_buyer"]]


def fetch_trades_mexc_futures() -> pd.DataFrame:
    """Fetch recent BTC_USDT perpetual futures trades from MEXC."""
    url = "https://contract.mexc.com/api/v1/contract/deals/BTC_USDT"
    params = {"limit": 1000}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("data"):
        return pd.DataFrame()

    df = pd.DataFrame(data["data"])
    df["timestamp"] = pd.to_datetime(df["t"].astype(float), unit="ms", utc=True)
    df["price"] = df["p"].astype(float)
    # MEXC futures: v is in contracts, 1 contract = 1 USDT → divide by price to get BTC
    df["size"] = df["v"].astype(float) / df["price"]
    df["is_buyer"] = df["T"] == 1  # T=1 buy, T=2 sell
    return df[["timestamp", "price", "size", "is_buyer"]]


# REST fetchers mapped to exchange name — used only for initial warmup
TRADES_EXCHANGES = {
    "mexc_spot":     (fetch_trades_mexc_spot,                                 "spot"),
    "bybit_spot":    (lambda: fetch_trades_bybit(category="spot"),            "spot"),
    "okx_spot":      (lambda: fetch_trades_okx(inst_id="BTC-USDT"),           "spot"),
    "gate_spot":     (fetch_trades_gate_spot,                                  "spot"),
    "coinbase":      (fetch_trades_coinbase,                                   "spot"),
    "kraken":        (fetch_trades_kraken,                                     "spot"),
    "bybit_futures": (lambda: fetch_trades_bybit(category="linear"),           "futures"),
    "okx_futures":   (lambda: fetch_trades_okx(inst_id="BTC-USDT-SWAP"),      "futures"),
    "gate_futures":  (fetch_trades_gate_futures,                               "futures"),
    "mexc_futures":  (fetch_trades_mexc_futures,                               "futures"),
}


# ---------------------------------------------------------------------------
# TRADE BUFFER — thread-safe in-memory accumulator
# ---------------------------------------------------------------------------

_trades_buffer: dict = {
    "mexc_spot":      pd.DataFrame(),
    "bybit_spot":     pd.DataFrame(),
    "okx_spot":       pd.DataFrame(),
    "gate_spot":      pd.DataFrame(),
    "coinbase":       pd.DataFrame(),
    "kraken":         pd.DataFrame(),
    "bybit_futures":  pd.DataFrame(),
    "okx_futures":    pd.DataFrame(),
    "gate_futures":   pd.DataFrame(),
    "mexc_futures":   pd.DataFrame(),
}

# One lock per exchange — protects concurrent access from WS threads vs flush thread
_buffer_locks: dict = {ex: threading.Lock() for ex in _trades_buffer}


def update_buffer(exchange: str, new_trades: pd.DataFrame) -> None:
    """Add new trades to buffer under lock, keep only last TRADES_BUFFER_MINUTES minutes."""
    if new_trades.empty:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=TRADES_BUFFER_MINUTES)
    with _buffer_locks[exchange]:
        combined = pd.concat([_trades_buffer[exchange], new_trades], ignore_index=True)
        combined = combined[combined["timestamp"] > cutoff].drop_duplicates(
            subset=["timestamp", "price", "size"]
        ).reset_index(drop=True)
        _trades_buffer[exchange] = combined


def aggregate_trades_to_candles(trades: pd.DataFrame, exchange: str) -> pd.DataFrame:
    """Resample raw trades into 5m OHLCV candles with taker buy volume."""
    if trades.empty:
        return pd.DataFrame()

    trades = trades.set_index("timestamp").sort_index()
    ohlcv = trades["price"].resample(CANDLE_INTERVAL).ohlc()
    volume = trades["size"].resample(CANDLE_INTERVAL).sum()
    taker_buy = (
        trades.loc[trades["is_buyer"], "size"]
        .resample(CANDLE_INTERVAL)
        .sum()
        .reindex(ohlcv.index)
        .fillna(0)
    )

    result = ohlcv.copy()
    result["volume"] = volume
    result["taker_buy_vol"] = taker_buy
    result["exchange"] = exchange
    result = result.dropna(subset=["open"]).reset_index()
    return result[["timestamp", "exchange", "open", "high", "low", "close", "volume", "taker_buy_vol"]]


def flush_buffer_to_parquet(exchange: str, market_type: str) -> None:
    """
    Aggregate completed 5m candles from buffer and write to parquet.
    Takes a snapshot of the buffer under lock, then processes outside the lock.
    Only flushes candles that are fully closed (not the currently forming one).
    """
    with _buffer_locks[exchange]:
        buf = _trades_buffer[exchange].copy()

    if buf.empty:
        return

    now = datetime.now(timezone.utc)
    current_candle_start = now.replace(second=0, microsecond=0)
    current_candle_start = current_candle_start - timedelta(minutes=current_candle_start.minute % 5)

    completed_trades = buf[buf["timestamp"] < current_candle_start]
    if completed_trades.empty:
        return

    candles = aggregate_trades_to_candles(completed_trades, exchange)
    if candles.empty:
        return

    filepath = DATA_DIR / f"btc_{market_type}_5m.parquet"
    n = upsert_parquet(filepath, candles)
    if n:
        log.info(f"[{exchange}] {market_type}: +{n} candles (ws)")


# ---------------------------------------------------------------------------
# WEBSOCKET — real-time trade streams for all non-Binance exchanges
# ---------------------------------------------------------------------------

# Exchange → market type, used by collect_cycle for flushing
_WS_EXCHANGE_MARKET: dict = {
    "mexc_spot":     "spot",
    "bybit_spot":    "spot",
    "okx_spot":      "spot",
    "gate_spot":     "spot",
    "coinbase":      "spot",
    "kraken":        "spot",
    "bybit_futures": "futures",
    "okx_futures":   "futures",
    "gate_futures":  "futures",
    "mexc_futures":  "futures",
}


# ------------------------------------------------------------------
# Per-exchange message parsers
# Each returns a DataFrame with columns: timestamp, price, size, is_buyer
# Returns empty DataFrame on any parse error or non-trade message.
# ------------------------------------------------------------------

def _parse_bybit(exchange: str, msg: str) -> pd.DataFrame:
    """
    Parse Bybit V5 publicTrade message.
    Size (v) is in BTC for both spot and linear perpetual.
    S: 'Buy' = taker buy, 'Sell' = taker sell.
    """
    data = json.loads(msg)
    if data.get("op") == "pong" or "data" not in data:
        return pd.DataFrame()
    trades = data["data"]
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "timestamp": pd.Timestamp(int(t["T"]), unit="ms", tz="UTC"),
        "price":     float(t["p"]),
        "size":      float(t["v"]),
        "is_buyer":  t["S"] == "Buy",
    } for t in trades])


def _parse_okx(exchange: str, msg: str) -> pd.DataFrame:
    """
    Parse OKX trades channel message.
    Spot (BTC-USDT): sz in BTC.
    Futures (BTC-USDT-SWAP): sz in contracts, 1 contract = 0.01 BTC.
    """
    if msg == "pong":
        return pd.DataFrame()
    data = json.loads(msg)
    if data.get("event") in ("subscribe", "error") or "data" not in data:
        return pd.DataFrame()
    is_futures = (exchange == "okx_futures")
    return pd.DataFrame([{
        "timestamp": pd.Timestamp(int(t["ts"]), unit="ms", tz="UTC"),
        "price":     float(t["px"]),
        "size":      float(t["sz"]) * (0.01 if is_futures else 1.0),
        "is_buyer":  t["side"] == "buy",
    } for t in data["data"]])


def _gate_ts(raw) -> pd.Timestamp:
    """
    Gate.io timestamp helper.
    Gate REST returns create_time_ms as float seconds (bug). Gate WS may return
    actual milliseconds. Handles both by checking magnitude.
    """
    val = float(raw)
    if val < 1e12:          # looks like seconds — convert to ms
        val *= 1000
    return pd.Timestamp(int(val), unit="ms", tz="UTC")


def _parse_gate_spot(msg: str) -> pd.DataFrame:
    """
    Parse Gate.io spot.trades WebSocket message.
    amount is in BTC. side: 'buy' = taker buy.
    """
    data = json.loads(msg)
    if data.get("event") != "update" or data.get("channel") != "spot.trades":
        return pd.DataFrame()
    r = data.get("result", {})
    if not r:
        return pd.DataFrame()
    ts_raw = r.get("create_time_ms") or r.get("create_time", 0)
    return pd.DataFrame([{
        "timestamp": _gate_ts(ts_raw),
        "price":     float(r["price"]),
        "size":      float(r["amount"]),
        "is_buyer":  r["side"] == "buy",
    }])


def _parse_gate_futures(msg: str) -> pd.DataFrame:
    """
    Parse Gate.io futures.trades WebSocket message.
    size positive = buy, negative = sell.
    1 contract = 1 USD → divide by price for BTC.
    """
    data = json.loads(msg)
    if data.get("event") != "update" or data.get("channel") != "futures.trades":
        return pd.DataFrame()
    results = data.get("result", [])
    if not results:
        return pd.DataFrame()
    rows = []
    for t in results:
        size_raw = float(t["size"])
        price    = float(t["price"])
        ts_raw   = t.get("create_time_ms") or t.get("create_time", 0)
        rows.append({
            "timestamp": _gate_ts(ts_raw),
            "price":     price,
            "size":      abs(size_raw) / price,
            "is_buyer":  size_raw > 0,
        })
    return pd.DataFrame(rows)


def _parse_mexc_spot(msg: str) -> pd.DataFrame:
    """
    Parse MEXC spot deals WebSocket message.
    v is in BTC. S: 1 = taker buy, 2 = taker sell.
    """
    data = json.loads(msg)
    deals = data.get("d", {}).get("deals", [])
    if not deals:
        return pd.DataFrame()
    return pd.DataFrame([{
        "timestamp": pd.Timestamp(int(t["t"]), unit="ms", tz="UTC"),
        "price":     float(t["p"]),
        "size":      float(t["v"]),
        "is_buyer":  int(t["S"]) == 1,
    } for t in deals])


def _parse_mexc_futures(msg: str) -> pd.DataFrame:
    """
    Parse MEXC futures push.deal WebSocket message.
    v in contracts, 1 contract = 1 USDT → divide by price for BTC.
    T: 1 = buy, 2 = sell.
    Message format: {"channel": "push.deal", "data": [{...}], "symbol": "BTC_USDT", "ts": ...}
    Note: data is a list directly, not a dict with a "deals" key.
    """
    data = json.loads(msg)
    if data.get("channel") != "push.deal":
        return pd.DataFrame()
    deals = data.get("data", [])
    if not isinstance(deals, list) or not deals:
        return pd.DataFrame()
    rows = []
    for t in deals:
        price = float(t["p"])
        rows.append({
            "timestamp": pd.Timestamp(int(t["t"]), unit="ms", tz="UTC"),
            "price":     price,
            "size":      float(t["v"]) / price,
            "is_buyer":  int(t["T"]) == 1,
        })
    return pd.DataFrame(rows)


def _parse_coinbase_adv(msg: str) -> pd.DataFrame:
    """
    Parse Coinbase Advanced Trade WebSocket market_trades message.
    IMPORTANT: Coinbase 'side' is the MAKER side, not the taker side.
    side='SELL' means maker sold → taker BOUGHT → is_buyer=True (+delta)
    side='BUY'  means maker bought → taker SOLD → is_buyer=False (-delta)
    Source: docs.cdp.coinbase.com/coinbase-app/advanced-trade-apis/websocket
    One message may contain multiple trades inside events[].trades[].
    Used as the sole Coinbase source (exchange='coinbase') — Advanced Trade has
    ~38x more activity than Exchange API and they share the same matching engine,
    so collecting both would double-count every trade.
    """
    data = json.loads(msg)
    if data.get("channel") != "market_trades":
        return pd.DataFrame()
    rows = []
    for event in data.get("events", []):
        for t in event.get("trades", []):
            ts = pd.Timestamp(t["time"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            rows.append({
                "timestamp": ts,
                "price":     float(t["price"]),
                "size":      float(t["size"]),
                "is_buyer":  t["side"] == "SELL",  # SELL=maker sold=taker bought
            })
    return pd.DataFrame(rows)


def _parse_kraken(msg: str) -> pd.DataFrame:
    """
    Parse Kraken V2 trade channel message.
    side: 'buy' = taker buy.
    """
    data = json.loads(msg)
    if data.get("channel") != "trade" or data.get("type") not in ("update", "snapshot"):
        return pd.DataFrame()
    trades = data.get("data", [])
    if not trades:
        return pd.DataFrame()
    rows = []
    for t in trades:
        ts = pd.Timestamp(t["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        rows.append({
            "timestamp": ts,
            "price":     float(t["price"]),
            "size":      float(t["qty"]),
            "is_buyer":  t["side"] == "buy",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# WsRunner — manages one WebSocket connection with full error handling
# ------------------------------------------------------------------

class WsRunner:
    """
    Manages a single persistent WebSocket connection for a trade stream.

    Reconnect logic:
      - Exponential backoff: 5s → 10s → 20s → 40s → 60s (max)
      - Backoff resets on every successful connect (on_open)
      - WebSocket-level ping frames (ping_interval=30s, ping_timeout=10s) detect
        dead TCP connections even when the server is silent

    Heartbeat (application-level ping):
      - Some exchanges disconnect after ~30s of client inactivity (Bybit, OKX, MEXC)
      - app_ping_msg is sent in a dedicated daemon thread at app_ping_interval seconds
      - Thread exits automatically when connection closes

    Stale detection:
      - last_trade_ts tracks when the last trade arrived
      - _health_monitor (external) calls ws.close() if stale > WS_STALE_S
        to force a reconnect even when the TCP connection looks alive
    """

    WS_STALE_S = 300  # seconds without a trade before forcing reconnect

    def __init__(
        self,
        exchange: str,
        ws_url: str,
        subscribe_fn: Callable[[], str],    # called on every connect, returns JSON to send
        parse_fn: Callable[[str], pd.DataFrame],
        app_ping_msg: Optional[str] = None,  # JSON string sent as heartbeat, or None
        app_ping_interval: int = 20,         # seconds between heartbeats
    ) -> None:
        self.exchange          = exchange
        self.ws_url            = ws_url
        self.subscribe_fn      = subscribe_fn
        self.parse_fn          = parse_fn
        self.app_ping_msg      = app_ping_msg
        self.app_ping_interval = app_ping_interval
        self.ws: Optional[websocket.WebSocketApp] = None
        self._connected        = False
        self._reconnect_delay  = 5
        self.last_trade_ts     = 0.0   # time.time() of last parsed trade

    # ------------------------------------------------------------------
    # WebSocketApp callbacks
    # ------------------------------------------------------------------

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        log.info(f"[{self.exchange}] WS connected")
        self._connected = True
        self._reconnect_delay = 5       # reset backoff on successful connect
        ws.send(self.subscribe_fn())
        if self.app_ping_msg:
            threading.Thread(
                target=self._ping_loop, daemon=True,
                name=f"ws-ping-{self.exchange}",
            ).start()

    def _on_message(self, ws: websocket.WebSocketApp, msg: str) -> None:
        try:
            trades = self.parse_fn(msg)
            if not trades.empty:
                self.last_trade_ts = time.time()
                update_buffer(self.exchange, trades)
        except Exception as e:
            log.debug(f"[{self.exchange}] parse error: {e}")

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        log.warning(f"[{self.exchange}] WS error: {error}")

    def _on_close(
        self,
        ws: websocket.WebSocketApp,
        close_status_code: Optional[int],
        close_msg: Optional[str],
    ) -> None:
        self._connected = False
        log.info(f"[{self.exchange}] WS closed (code={close_status_code})")

    # ------------------------------------------------------------------
    # Application-level ping loop
    # ------------------------------------------------------------------

    def _ping_loop(self) -> None:
        """Send application-level heartbeat while connected."""
        while self._connected:
            time.sleep(self.app_ping_interval)
            if self._connected and self.ws:
                try:
                    self.ws.send(self.app_ping_msg)
                except Exception as e:
                    log.debug(f"[{self.exchange}] heartbeat error: {e}")

    # ------------------------------------------------------------------
    # Reconnect loop — runs for the lifetime of the daemon thread
    # ------------------------------------------------------------------

    def _run_with_reconnect(self) -> None:
        while True:
            try:
                log.info(f"[{self.exchange}] Connecting...")
                self.ws = websocket.WebSocketApp(
                    self.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # ping_interval sends WebSocket-level ping frames to detect dead connections
                self.ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                log.error(f"[{self.exchange}] WS run exception: {e}")
            finally:
                self._connected = False

            log.info(f"[{self.exchange}] Reconnecting in {self._reconnect_delay}s...")
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, 60)

    def start(self) -> None:
        """Start the WebSocket connection in a daemon thread."""
        threading.Thread(
            target=self._run_with_reconnect, daemon=True,
            name=f"ws-{self.exchange}",
        ).start()


# ------------------------------------------------------------------
# WS runner factory
# ------------------------------------------------------------------

def _static_sub(payload: dict) -> Callable[[], str]:
    """Subscribe factory for static payloads (no dynamic fields)."""
    s = json.dumps(payload)
    return lambda: s


def _gate_sub(channel: str, payload: list) -> Callable[[], str]:
    """Subscribe factory for Gate.io (requires fresh Unix timestamp on each connect)."""
    return lambda: json.dumps({
        "time":    int(time.time()),
        "channel": channel,
        "event":   "subscribe",
        "payload": payload,
    })


def _build_ws_runners() -> List[WsRunner]:
    """Instantiate one WsRunner per non-Binance exchange."""
    return [
        # ---- Bybit spot ----
        # publicTrade.BTCUSDT: size (v) in BTC, S: 'Buy'/'Sell' is taker side
        # Bybit disconnects after 30s of inactivity → heartbeat every 20s
        WsRunner(
            exchange="bybit_spot",
            ws_url="wss://stream.bybit.com/v5/public/spot",
            subscribe_fn=_static_sub({"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}),
            parse_fn=lambda msg: _parse_bybit("bybit_spot", msg),
            app_ping_msg=json.dumps({"op": "ping"}),
            app_ping_interval=20,
        ),
        # ---- Bybit futures (linear perpetual) ----
        # Same format as spot; size (v) is in BTC for linear BTCUSDT
        WsRunner(
            exchange="bybit_futures",
            ws_url="wss://stream.bybit.com/v5/public/linear",
            subscribe_fn=_static_sub({"op": "subscribe", "args": ["publicTrade.BTCUSDT"]}),
            parse_fn=lambda msg: _parse_bybit("bybit_futures", msg),
            app_ping_msg=json.dumps({"op": "ping"}),
            app_ping_interval=20,
        ),
        # ---- OKX spot ----
        # sz in BTC. OKX expects string "ping" every ≤30s or it disconnects
        WsRunner(
            exchange="okx_spot",
            ws_url="wss://ws.okx.com:8443/ws/v5/public",
            subscribe_fn=_static_sub({"op": "subscribe", "args": [{"channel": "trades", "instId": "BTC-USDT"}]}),
            parse_fn=lambda msg: _parse_okx("okx_spot", msg),
            app_ping_msg="ping",
            app_ping_interval=25,
        ),
        # ---- OKX futures (BTC-USDT-SWAP) ----
        # sz in contracts, 1 contract = 0.01 BTC
        WsRunner(
            exchange="okx_futures",
            ws_url="wss://ws.okx.com:8443/ws/v5/public",
            subscribe_fn=_static_sub({"op": "subscribe", "args": [{"channel": "trades", "instId": "BTC-USDT-SWAP"}]}),
            parse_fn=lambda msg: _parse_okx("okx_futures", msg),
            app_ping_msg="ping",
            app_ping_interval=25,
        ),
        # ---- Gate spot ----
        # amount in BTC. Gate uses WS-level ping frames — no app heartbeat needed
        WsRunner(
            exchange="gate_spot",
            ws_url="wss://api.gateio.ws/ws/v4/",
            subscribe_fn=_gate_sub("spot.trades", ["BTC_USDT"]),
            parse_fn=_parse_gate_spot,
            app_ping_msg=None,
        ),
        # ---- Gate futures (BTC_USDT USDT-margined perp) ----
        # size positive=buy, negative=sell; 1 contract = 1 USD → /price = BTC
        WsRunner(
            exchange="gate_futures",
            ws_url="wss://fx-ws.gateio.ws/v4/ws/usdt",
            subscribe_fn=_gate_sub("futures.trades", ["BTC_USDT"]),
            parse_fn=_parse_gate_futures,
            app_ping_msg=None,
        ),
        # ---- MEXC spot ----
        # v in BTC. S: 1=buy, 2=sell. Uses WS-level frames — no app heartbeat needed
        WsRunner(
            exchange="mexc_spot",
            ws_url="wss://wbs.mexc.com/ws",
            subscribe_fn=_static_sub({"method": "SUBSCRIPTION", "params": ["spot@public.deals.v3.api@BTCUSDT"]}),
            parse_fn=_parse_mexc_spot,
            app_ping_msg=None,
        ),
        # ---- MEXC futures (BTC_USDT) ----
        # v in contracts, 1 contract = 1 USDT → /price = BTC. T: 1=buy, 2=sell
        # MEXC contract WS requires JSON ping every ≤15s
        WsRunner(
            exchange="mexc_futures",
            ws_url="wss://contract.mexc.com/edge",
            subscribe_fn=_static_sub({"method": "sub.deal", "param": {"symbol": "BTC_USDT"}}),
            parse_fn=_parse_mexc_futures,
            app_ping_msg=json.dumps({"method": "ping"}),
            app_ping_interval=15,
        ),
        # ---- Coinbase Advanced Trade ----
        # Sole Coinbase source. Advanced Trade and Exchange API share the same matching
        # engine (identical trade_ids confirmed) — using both would double-count every trade.
        # Advanced Trade has ~38x more activity (complete feed); Exchange API is a subset.
        # side: 'BUY'/'SELL' is the taker side.
        WsRunner(
            exchange="coinbase",
            ws_url="wss://advanced-trade-ws.coinbase.com",
            subscribe_fn=_static_sub({"type": "subscribe", "product_ids": ["BTC-USD"], "channel": "market_trades"}),
            parse_fn=_parse_coinbase_adv,
            app_ping_msg=None,
        ),
        # ---- Kraken V2 ----
        # qty in BTC. side: 'buy'/'sell' is taker side. Kraken sends heartbeats — no app ping needed
        WsRunner(
            exchange="kraken",
            ws_url="wss://ws.kraken.com/v2",
            subscribe_fn=_static_sub({"method": "subscribe", "params": {"channel": "trade", "symbol": ["BTC/USD"]}}),
            parse_fn=_parse_kraken,
            app_ping_msg=None,
        ),
    ]


def _health_monitor(runners: List[WsRunner]) -> None:
    """
    Background thread: checks each WsRunner every 60s.
    Forces reconnect by closing the socket if no trades received for WS_STALE_S.
    This handles the case where the TCP connection is alive but the exchange
    stopped sending data (e.g. dropped subscription without closing the socket).
    """
    while True:
        time.sleep(60)
        now = time.time()
        for runner in runners:
            if not runner._connected:
                continue
            if runner.last_trade_ts > 0 and now - runner.last_trade_ts > runner.WS_STALE_S:
                log.warning(
                    f"[{runner.exchange}] No trades for "
                    f"{now - runner.last_trade_ts:.0f}s — forcing reconnect"
                )
                try:
                    runner.ws.close()
                except Exception:
                    pass


def start_websockets() -> None:
    """Start all WebSocket streams and the health monitor."""
    runners = _build_ws_runners()
    for runner in runners:
        runner.start()
        time.sleep(0.2)  # stagger connections slightly
    threading.Thread(
        target=_health_monitor, args=(runners,),
        daemon=True, name="ws-health",
    ).start()
    log.info(f"[ws] Started {len(runners)} WebSocket streams + health monitor")


# ---------------------------------------------------------------------------
# OPEN INTEREST — Binance Futures
# ---------------------------------------------------------------------------

def fetch_oi_snapshot_binance() -> dict:
    """Fetch current open interest from Binance Futures."""
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    resp = requests.get(url, params={"symbol": "BTCUSDT"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return {
        "timestamp": datetime.now(timezone.utc).replace(second=0, microsecond=0),
        "exchange":  "binance_futures",
        "oi_value":  float(data["openInterest"]),
    }


def backfill_oi_binance() -> None:
    """
    Backfill OI history using Binance 5m historical endpoint.
    On first run: 5 days. On restart: only the gap since last stored snapshot.
    """
    filepath = DATA_DIR / "btc_oi_1m.parquet"
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    start_ms = last_stored_ts(filepath, "binance_futures")
    all_dfs = []

    log.info(f"[binance_futures] Backfilling OI from {ms_to_dt(start_ms).strftime('%Y-%m-%d %H:%M')}...")
    while start_ms < now_ms():
        params = {"symbol": "BTCUSDT", "period": "5m", "limit": 500, "startTime": start_ms}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms", utc=True)
        df["exchange"] = "binance_futures"
        df["oi_value"] = df["sumOpenInterest"].astype(float)
        all_dfs.append(df[["timestamp", "exchange", "oi_value"]])

        start_ms = int(df["timestamp"].max().timestamp() * 1000) + 5 * 60 * 1000
        if len(data) < 500:
            break
        time.sleep(0.2)

    if all_dfs:
        result = pd.concat(all_dfs, ignore_index=True)
        upsert_parquet(filepath, result)
        log.info(f"[binance_futures] OI backfill done: {len(result)} snapshots")


def update_oi_binance() -> None:
    """Poll current OI and append to parquet."""
    filepath = DATA_DIR / "btc_oi_1m.parquet"
    try:
        snapshot = fetch_oi_snapshot_binance()
        upsert_parquet(filepath, pd.DataFrame([snapshot]))
    except Exception as e:
        log.error(f"[binance_futures] OI update error: {e}")


# ---------------------------------------------------------------------------
# COLLECTION CYCLE
# ---------------------------------------------------------------------------

def collect_cycle() -> None:
    """
    One full data collection cycle (runs every UPDATE_INTERVAL_S seconds).
    Binance klines are fetched via REST. All other exchanges feed through
    WebSocket streams — this cycle only flushes their accumulated buffers.
    """

    # 1. Native klines (Binance spot)
    for exchange in ["binance"]:
        try:
            update_spot_native(exchange)
        except Exception as e:
            log.error(f"[{exchange}] spot klines error: {e}")

    # 2. Native klines (Binance futures)
    try:
        update_futures_binance()
    except Exception as e:
        log.error(f"[binance_futures] klines error: {e}")

    # 3. Flush WebSocket trade buffers to parquet
    for exchange, market_type in _WS_EXCHANGE_MARKET.items():
        try:
            flush_buffer_to_parquet(exchange, market_type)
        except Exception as e:
            log.error(f"[{exchange}] flush error: {e}")

    # 4. OI snapshot
    update_oi_binance()


# ---------------------------------------------------------------------------
# BACKFILL
# ---------------------------------------------------------------------------

def run_backfill() -> None:
    """
    One-time backfill on startup.
    Native klines: full 5-day history (Binance spot + futures).
    OI: 5-day history via Binance 5m historical endpoint.
    WS exchanges: seeded with REST snapshot (~last 1000 trades) to warm up
    the buffer before WebSocket connections come online.
    """
    log.info("=" * 50)
    log.info("Starting backfill...")
    log.info("=" * 50)

    for exchange in ["binance"]:
        try:
            backfill_spot_native(exchange)
        except Exception as e:
            log.error(f"Backfill [{exchange}] failed: {e}")
        time.sleep(0.5)

    try:
        backfill_futures_binance()
    except Exception as e:
        log.error(f"Backfill [binance_futures] failed: {e}")

    try:
        backfill_oi_binance()
    except Exception as e:
        log.error(f"Backfill [binance_futures OI] failed: {e}")

    # Seed WS exchange buffers with recent REST snapshot
    log.info("Seeding trade buffers with REST snapshot (WS warmup)...")
    for exchange, (fetcher, market_type) in TRADES_EXCHANGES.items():
        try:
            new_trades = fetcher()
            update_buffer(exchange, new_trades)
            log.info(f"[{exchange}] warmup: {len(new_trades)} trades")
        except Exception as e:
            log.error(f"[{exchange}] warmup failed: {e}")
        time.sleep(0.3)

    log.info("=" * 50)
    log.info("Backfill complete.")
    log.info("=" * 50)


# ---------------------------------------------------------------------------
# DIVERGENCE ALERTS
# ---------------------------------------------------------------------------

# TFs to check for divergences on every collection cycle
ALERT_TIMEFRAMES = ["5m", "15m", "30m", "1h"]
MIN_DIV_SCORE    = 0.50  # minimum persistence [0–1] to trigger an alert — only empirically justified filter

# State file — persists last-alerted candle timestamps across restarts
ALERT_STATE_FILE = DATA_DIR / "alert_state.json"

_SIGNAL_EMOJI = {
    "SELLING EXHAUSTION": "🔵",
    "SELLING ABSORPTION": "🟢",
    "BUYING EXHAUSTION":  "🟠",
    "BUYING ABSORPTION":  "🔴"
}

# All spot exchanges used for CVD aggregation (mirrors app.py default)
_ALL_SPOT_EXCHANGES = [
    "binance", "mexc", "bybit_spot", "okx_spot",
    "gate_spot", "coinbase", "kraken",
]


def _load_alert_state() -> dict:
    """Load persisted alert state from disk. Returns empty state if not found."""
    if ALERT_STATE_FILE.exists():
        try:
            return json.loads(ALERT_STATE_FILE.read_text())
        except Exception:
            pass
    return {tf: {"low": None, "high": None} for tf in ALERT_TIMEFRAMES}


def _save_alert_state(state: dict) -> None:
    """Persist alert state to disk."""
    try:
        ALERT_STATE_FILE.write_text(json.dumps(state))
    except Exception:
        pass


def _format_alert_message(
    signal_data: dict,
    timeframes: list,
    current_price: float,
    oi_df: pd.DataFrame,
) -> str:
    """
    Format Telegram message text for a divergence alert.
    timeframes: list of TF strings sorted highest-first, e.g. ["1h", "15m", "5m"]
    oi_df: trimmed OI DataFrame used to compute OI change between pivots.
    """
    sig    = signal_data["signal"]
    emoji  = _SIGNAL_EMOJI.get(sig, "⚪")
    p_from = signal_data["price_from"]
    p_to   = signal_data["price_to"]
    c_from = signal_data["cvd_from"]
    c_to   = signal_data["cvd_to"]
    ts     = signal_data["timestamp"]

    p_arrow = "↑" if p_to >= p_from else "↓"
    c_arrow = "↑" if c_to >= c_from else "↓"
    p_pct   = (p_to - p_from) / p_from * 100 if p_from else 0.0
    from zoneinfo import ZoneInfo
    ts_str  = pd.Timestamp(ts).tz_convert(ZoneInfo("Europe/Warsaw")).strftime("%H:%M (UTC+2)")
    tf_str  = "".join(f"[{tf}]" for tf in timeframes)

    net_delta   = c_to - c_from
    p_delta_usd = p_to - p_from
    persistence      = signal_data.get("persistence")
    price_atr_ratio  = signal_data.get("price_atr_ratio")
    cvd_sigma        = signal_data.get("cvd_sigma")
    window_bars      = signal_data.get("window_bars")
    futures_cvd_delta = signal_data.get("futures_cvd_delta")

    dims = []
    if persistence  is not None: dims.append(f"Persistence: {persistence*100:.0f}%")
    if window_bars  is not None: dims.append(f"Window: {window_bars}bars")
    dims_line = "  ·  ".join(dims)

    mag_parts = []
    if price_atr_ratio is not None: mag_parts.append(f"P.Move: {price_atr_ratio:.1f}×ATR")
    if cvd_sigma       is not None: mag_parts.append(f"CVD: {cvd_sigma:.1f}σ")
    mag_line = "  ·  ".join(mag_parts)

    fut_line = ""
    if futures_cvd_delta is not None:
        fut_arrow = "↑" if futures_cvd_delta >= 0 else "↓"
        fut_line  = f"Fut.CVD: {fut_arrow} {futures_cvd_delta:+,.0f} BTC\n"

    score_line = (
        f"{dims_line}\n"
        f"{mag_line}\n"
        f"{fut_line}\n"
        if dims_line else "\n"
    )

    # OI context: compare OI at pivot A vs pivot B timestamps
    oi_str = ""
    pivot_from_ts = signal_data.get("pivot_from_ts")
    if not oi_df.empty and pivot_from_ts is not None:
        try:
            oi_sorted = oi_df.set_index("timestamp").sort_index()
            oi_a = float(oi_sorted["close"].asof(pd.Timestamp(pivot_from_ts)))
            oi_b = float(oi_sorted["close"].asof(pd.Timestamp(ts)))
            if pd.notna(oi_a) and pd.notna(oi_b) and oi_a > 0:
                oi_delta = oi_b - oi_a
                oi_pct   = oi_delta / oi_a * 100
                oi_arrow = "↑" if oi_delta >= 0 else "↓"
                oi_label = "new positions" if oi_delta >= 0 else "covering"
                oi_str   = f"OI:      {oi_arrow} {oi_delta:+,.0f} BTC ({oi_pct:+.2f}%) · {oi_label}\n"
        except Exception:
            pass

    return (
        f"{emoji} <b>{sig}</b> {tf_str}\n"
        f"BTC/USDT @ {current_price:,.2f}\n"
        f"Price: {p_arrow} {p_delta_usd:+,.2f} USD  [{p_pct:+.2f}%]\n"
        f"CVD : {c_arrow} {net_delta:+,.0f} BTC\n"
        f"{oi_str}"
        f"\n"
        f"{score_line}"
        f"{ts_str}"
    )


def _build_alert_image(
    price_df: pd.DataFrame,
    cvd_spot_df: pd.DataFrame,
    cvd_futures_df: pd.DataFrame,
    oi_df: pd.DataFrame,
    interval_str: str,
) -> Optional[bytes]:
    """Render 540×960 PNG (9:16) of the last ALERT_CANDLES. Returns None if kaleido missing."""
    try:
        import kaleido  # noqa: F401
        from analysis import build_alert_figure, trim_to_candles, ALERT_CANDLES
    except ImportError:
        return None

    p  = trim_to_candles(price_df,       ALERT_CANDLES)
    cs = trim_to_candles(cvd_spot_df,    ALERT_CANDLES)
    cf = trim_to_candles(cvd_futures_df, ALERT_CANDLES)
    oi = trim_to_candles(oi_df,          ALERT_CANDLES)

    fig = build_alert_figure(p, cs, cf, oi, interval_str=interval_str)
    return fig.to_image(format="png", width=540, height=960)


def _send_telegram(text: str, image_bytes: Optional[bytes]) -> None:
    """Send photo (with caption) or text message to Telegram."""
    if not _TELEGRAM_TOKEN or not _TELEGRAM_CHAT_ID:
        return
    base = f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}"
    try:
        if image_bytes:
            requests.post(
                f"{base}/sendPhoto",
                data={"chat_id": _TELEGRAM_CHAT_ID, "caption": text, "parse_mode": "HTML"},
                files={"photo": ("alert.png", image_bytes, "image/png")},
                timeout=20,
            )
        else:
            requests.post(
                f"{base}/sendMessage",
                json={"chat_id": _TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")


def _append_signal_log(signal_data: dict, timeframes: list, current_price: float) -> None:
    """Append one row to the signal log CSV. Creates file with header if it doesn't exist."""
    row = {
        "timestamp":         pd.Timestamp(signal_data["timestamp"]).isoformat(),
        "sent_at":           pd.Timestamp.utcnow().isoformat(),
        "timeframes":        ",".join(timeframes),
        "signal":            signal_data["signal"],
        "price_from":        signal_data.get("price_from"),
        "price_to":          signal_data.get("price_to"),
        "cvd_from":          signal_data.get("cvd_from"),
        "cvd_to":            signal_data.get("cvd_to"),
        "price_move_pct":    signal_data.get("price_move_pct"),
        "cvd_move_pct":      signal_data.get("cvd_move_pct"),
        "persistence":       signal_data.get("persistence"),
        "price_atr_ratio":   signal_data.get("price_atr_ratio"),
        "cvd_sigma":         signal_data.get("cvd_sigma"),
        "window_bars":       signal_data.get("window_bars"),
        "oi_delta_btc":      signal_data.get("oi_delta_btc"),
        "futures_cvd_delta": signal_data.get("futures_cvd_delta"),
        "btc_price":         current_price,
        "cvd_mode":          "line",
        "pivot_from_ts":     pd.Timestamp(signal_data["pivot_from_ts"]).isoformat() if signal_data.get("pivot_from_ts") is not None else None,
    }
    df_row = pd.DataFrame([row])
    write_header = not SIGNAL_LOG_FILE.exists()
    df_row.to_csv(SIGNAL_LOG_FILE, mode="a", header=write_header, index=False)


def _alert_worker(
    entries: list,
    price_df: pd.DataFrame,
    cvd_spot_df: pd.DataFrame,
    cvd_futures_df: pd.DataFrame,
    oi_df: pd.DataFrame,
    top_tf: str,
) -> None:
    """Background thread: log each TF separately, render image, send Telegram."""
    current_price = float(price_df["close"].iloc[-1]) if not price_df.empty else 0.0

    # Log each timeframe as an individual row so per-TF metrics are preserved
    for tf, signal_data, *_ in entries:
        _append_signal_log(signal_data, [tf], current_price)

    top_signal_data = entries[0][1]
    timeframes      = [e[0] for e in entries]
    image_bytes     = _build_alert_image(price_df, cvd_spot_df, cvd_futures_df, oi_df, top_tf)
    text            = _format_alert_message(top_signal_data, timeframes, current_price, oi_df)
    _send_telegram(text, image_bytes)
    log.info(f"[alert] {' · '.join(timeframes)} · {top_signal_data['signal']} → Telegram sent")


# TF priority for sorting (highest first)
_TF_PRIORITY = {"1h": 4, "30m": 3, "15m": 2, "5m": 1}


def _enrich_signal_with_market_context(
    signal_data:    dict,
    oi_df:          pd.DataFrame,
    cvd_futures_df: pd.DataFrame,
) -> None:
    """
    Enrich signal_data with raw market context dimensions for backtesting.

    Adds:
      oi_delta_pct    — % change in OI between pivot timestamps (raw)
      futures_cvd_delta — BTC change in futures CVD between pivot timestamps (raw)

    Does NOT compute a composite score — all values are logged as-is so that
    weights and thresholds can be determined empirically after data collection.
    Modifies signal_data in-place.
    """
    pivot_from_ts = signal_data.get("pivot_from_ts")
    ts            = signal_data.get("timestamp")

    # --- OI delta (absolute BTC) ---
    oi_delta_btc = None
    if not oi_df.empty and pivot_from_ts is not None and ts is not None:
        try:
            oi_sorted = oi_df.set_index("timestamp").sort_index()
            oi_a = float(oi_sorted["close"].asof(pd.Timestamp(pivot_from_ts)))
            oi_b = float(oi_sorted["close"].asof(pd.Timestamp(ts)))
            if pd.notna(oi_a) and pd.notna(oi_b):
                oi_delta_btc = oi_b - oi_a
        except Exception:
            pass

    # --- Futures CVD delta ---
    futures_cvd_delta = None
    if not cvd_futures_df.empty and pivot_from_ts is not None and ts is not None:
        try:
            fut_sorted = cvd_futures_df.set_index("timestamp").sort_index()
            fut_a = float(fut_sorted["cvd_close"].asof(pd.Timestamp(pivot_from_ts)))
            fut_b = float(fut_sorted["cvd_close"].asof(pd.Timestamp(ts)))
            if pd.notna(fut_a) and pd.notna(fut_b):
                futures_cvd_delta = fut_b - fut_a
        except Exception:
            pass

    signal_data["oi_delta_btc"]      = oi_delta_btc
    signal_data["futures_cvd_delta"] = futures_cvd_delta


def check_and_alert(state: dict) -> None:
    """
    Check all ALERT_TIMEFRAMES for active divergence signals.

    Collects all new signals in this cycle, groups them by signal type,
    then sends one merged Telegram alert per type (highest TF image + all TFs listed).
    State is updated in-place and must be persisted by the caller.
    """
    from analysis import (
        INTERVAL_MAP, DISPLAY_CANDLES,
        resample_klines, compute_cvd, compute_oi_ohlc,
        trim_to_candles, reset_cvd_origin, get_price_df, detect_spot_signals,
    )

    spot_raw    = load_parquet(DATA_DIR / "btc_spot_5m.parquet")
    futures_raw = load_parquet(DATA_DIR / "btc_futures_5m.parquet")
    oi_raw      = load_parquet(DATA_DIR / "btc_oi_1m.parquet")

    if spot_raw.empty:
        return

    futures_exchanges = (
        list(futures_raw["exchange"].unique()) if not futures_raw.empty else []
    )

    # Collect new signals this cycle:
    # signal_name -> list of (tf, signal_data, price_df, cvd_spot_df, cvd_futures_df, oi_df)
    pending: dict = {}

    for tf in ALERT_TIMEFRAMES:
        pandas_interval = INTERVAL_MAP[tf]

        spot_rs    = resample_klines(spot_raw,    pandas_interval)
        futures_rs = resample_klines(futures_raw, pandas_interval)

        price_df       = trim_to_candles(get_price_df(spot_rs),                       DISPLAY_CANDLES)
        cvd_spot_df    = reset_cvd_origin(trim_to_candles(compute_cvd(spot_rs,    _ALL_SPOT_EXCHANGES), DISPLAY_CANDLES))
        cvd_futures_df = reset_cvd_origin(trim_to_candles(compute_cvd(futures_rs, futures_exchanges),   DISPLAY_CANDLES))
        oi_df          = trim_to_candles(compute_oi_ohlc(oi_raw, pandas_interval),     DISPLAY_CANDLES)

        low_data, high_data = detect_spot_signals(price_df, cvd_spot_df)

        if tf not in state:
            state[tf] = {"low": None, "high": None}

        for key, signal_data in [("low", low_data), ("high", high_data)]:
            if signal_data is None:
                continue
            ts_str = pd.Timestamp(signal_data["timestamp"]).isoformat()
            if ts_str == state[tf].get(key):
                continue  # already alerted for this candle
            _enrich_signal_with_market_context(signal_data, oi_df, cvd_futures_df)
            state[tf][key] = ts_str
            sig_name = signal_data["signal"]
            if sig_name not in pending:
                pending[sig_name] = []
            pending[sig_name].append((tf, signal_data, price_df, cvd_spot_df, cvd_futures_df, oi_df))

    # One alert per signal type — highest TF provides image + price data
    for sig_name, entries in pending.items():
        entries.sort(key=lambda x: _TF_PRIORITY.get(x[0], 0), reverse=True)
        timeframes = [e[0] for e in entries]
        top_tf, _, price_df, cvd_spot_df, cvd_futures_df, oi_df = entries[0]
        threading.Thread(
            target=_alert_worker,
            args=(entries, price_df, cvd_spot_df, cvd_futures_df, oi_df, top_tf),
            daemon=True,
        ).start()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def start() -> None:
    """Backfill historical data, start WebSocket streams, then run the collection loop."""
    DATA_DIR.mkdir(exist_ok=True)
    run_backfill()
    start_websockets()

    alert_state = _load_alert_state()
    log.info(
        f"Collector running — update interval: {UPDATE_INTERVAL_S}s, "
        f"{len(_WS_EXCHANGE_MARKET)} WS streams active"
    )

    while True:
        try:
            collect_cycle()
        except Exception as e:
            log.error(f"Collection cycle failed: {e}")

        try:
            check_and_alert(alert_state)
            _save_alert_state(alert_state)
        except Exception as e:
            log.error(f"Alert check failed: {e}")

        try:
            from outcome_tracker import check_outcomes
            check_outcomes()
        except Exception as e:
            log.error(f"Outcome tracking failed: {e}")

        time.sleep(UPDATE_INTERVAL_S)


if __name__ == "__main__":
    start()
