"""
collector.py — Data collector for aggregated CVD scanner.

Fetches OHLC + taker buy volume from multiple exchanges and stores in Parquet.
Runs continuously, updating every 60 seconds.

Output files:
  data/btc_spot_5m.parquet     columns: timestamp, exchange, open, high, low, close, volume, taker_buy_vol
  data/btc_futures_5m.parquet  columns: timestamp, exchange, open, high, low, close, volume, taker_buy_vol
  data/btc_oi_1m.parquet       columns: timestamp, exchange, oi_value

Spot exchanges:
  Native taker buy (klines):        binance
  REST server-side aggregation:     okx_spot
  REST polling (trade streams):     bybit_spot, gate_spot, coinbase, kraken

Futures exchanges:
  Native taker buy (klines):        binance_futures
  REST server-side aggregation:     okx_futures
  REST polling (trade streams):     bybit_futures, gate_futures

OI: binance_futures (polled every 60s, backfilled via 5m historical endpoint)

Polling design:
  - One PollingRunner thread per exchange (daemon)
  - Calls fetch_fn every POLL_INTERVAL_S seconds
  - fetch_fn is stateful (closure with last_ts / cursor)
  - OKX uses server-side 5m candle aggregation (rubik/stat/taker-volume + market/candles)
"""

import json
import os
import threading
import time
import logging

import requests
import pandas as pd
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
# OKX — REST SERVER-SIDE AGGREGATION
# rubik/stat/taker-volume gives buy/sell BTC per 5m candle (server-computed).
# market/candles gives OHLCV. Both are merged on timestamp (inner join).
# ---------------------------------------------------------------------------

def fetch_okx_with_taker(
    exchange: str,
    inst_id: str,
    ccy: str,
    inst_type: str,
    limit: int = 10,
) -> pd.DataFrame:
    """Fetch recent OKX candles with taker buy volume via server-side aggregation.

    Uses rubik/stat/taker-volume (buy/sell BTC per 5m) merged with market/candles (OHLCV).
    Inner join on timestamp — if either endpoint fails or has no overlap, returns empty DataFrame.
    volume = buy_vol + sell_vol from rubik (already in BTC for both SPOT and CONTRACTS).
    """
    try:
        rubik_resp = requests.get(
            "https://www.okx.com/api/v5/rubik/stat/taker-volume",
            params={"ccy": ccy, "instType": inst_type, "period": "5m", "limit": limit},
            timeout=10,
        )
        rubik_resp.raise_for_status()
        rubik_data = rubik_resp.json()
        if rubik_data.get("code") != "0" or not rubik_data.get("data"):
            log.warning(f"[{exchange}] rubik error: {rubik_data.get('msg')}")
            return pd.DataFrame()
    except Exception as e:
        log.warning(f"[{exchange}] rubik fetch failed: {e}")
        return pd.DataFrame()

    try:
        candles_resp = requests.get(
            "https://www.okx.com/api/v5/market/candles",
            params={"instId": inst_id, "bar": "5m", "limit": limit},
            timeout=10,
        )
        candles_resp.raise_for_status()
        candles_data = candles_resp.json()
        if candles_data.get("code") != "0" or not candles_data.get("data"):
            log.warning(f"[{exchange}] candles error: {candles_data.get('msg')}")
            return pd.DataFrame()
    except Exception as e:
        log.warning(f"[{exchange}] candles fetch failed: {e}")
        return pd.DataFrame()

    # rubik: [[ts_ms, buy_vol, sell_vol], ...] newest first
    rubik_rows = {}
    for row in rubik_data["data"]:
        ts_ms = int(row[0])
        rubik_rows[ts_ms] = {
            "taker_buy_vol": float(row[1]),
            "taker_sell_vol": float(row[2]),
        }

    # candles: [[ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm], ...] newest first
    candle_rows = {}
    for row in candles_data["data"]:
        ts_ms = int(row[0])
        candle_rows[ts_ms] = {
            "open":  float(row[1]),
            "high":  float(row[2]),
            "low":   float(row[3]),
            "close": float(row[4]),
        }

    # inner join on common timestamps
    common_ts = set(rubik_rows.keys()) & set(candle_rows.keys())
    if not common_ts:
        log.debug(f"[{exchange}] no overlapping timestamps between rubik and candles")
        return pd.DataFrame()

    records = []
    for ts_ms in sorted(common_ts):
        r = rubik_rows[ts_ms]
        c = candle_rows[ts_ms]
        records.append({
            "timestamp":     pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
            "exchange":      exchange,
            "open":          c["open"],
            "high":          c["high"],
            "low":           c["low"],
            "close":         c["close"],
            "volume":        r["taker_buy_vol"] + r["taker_sell_vol"],
            "taker_buy_vol": r["taker_buy_vol"],
        })

    return pd.DataFrame(records)[
        ["timestamp", "exchange", "open", "high", "low", "close", "volume", "taker_buy_vol"]
    ]


def backfill_okx(
    exchange: str,
    inst_id: str,
    ccy: str,
    inst_type: str,
) -> None:
    """Backfill OKX candles with taker volume from last_stored_ts to now.

    Candles: page backwards using after=<oldest_ts> until reaching start_ms.
    Rubik: fetch forward in chunks using begin/end parameters (100 candles = 8.3h per chunk).
    Merges on common timestamps (inner join).
    """
    filepath = DATA_DIR / ("btc_spot_5m.parquet" if "spot" in exchange else "btc_futures_5m.parquet")
    start_ms = last_stored_ts(filepath, exchange)
    end_ms = now_ms()

    log.info(f"[{exchange}] Backfilling OKX from {ms_to_dt(start_ms).strftime('%Y-%m-%d %H:%M')}...")

    # --- Fetch all candles paging backwards ---
    candle_rows: dict = {}
    after_ts = end_ms
    while True:
        try:
            resp = requests.get(
                "https://www.okx.com/api/v5/market/candles",
                params={"instId": inst_id, "bar": "5m", "limit": 300, "after": after_ts},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"[{exchange}] candles page failed: {e}")
            break

        if data.get("code") != "0" or not data.get("data"):
            break

        batch = data["data"]
        for row in batch:
            ts_ms = int(row[0])
            if ts_ms < start_ms:
                continue
            candle_rows[ts_ms] = {
                "open":  float(row[1]),
                "high":  float(row[2]),
                "low":   float(row[3]),
                "close": float(row[4]),
            }

        oldest_in_batch = int(batch[-1][0])
        if oldest_in_batch <= start_ms or len(batch) < 300:
            break
        after_ts = oldest_in_batch
        time.sleep(0.2)

    if not candle_rows:
        log.info(f"[{exchange}] No candles to backfill")
        return

    # --- Fetch rubik in forward chunks (100 candles = 500 min per chunk) ---
    CHUNK_MS = 100 * INTERVAL_5M_MS  # 500 minutes in ms
    rubik_rows: dict = {}
    chunk_start = start_ms
    while chunk_start < end_ms:
        chunk_end = min(chunk_start + CHUNK_MS, end_ms)
        try:
            resp = requests.get(
                "https://www.okx.com/api/v5/rubik/stat/taker-volume",
                params={
                    "ccy": ccy,
                    "instType": inst_type,
                    "period": "5m",
                    "begin": chunk_start,
                    "end": chunk_end,
                    "limit": 100,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.warning(f"[{exchange}] rubik chunk failed: {e}")
            chunk_start = chunk_end
            continue

        if data.get("code") == "0" and data.get("data"):
            for row in data["data"]:
                ts_ms = int(row[0])
                rubik_rows[ts_ms] = {
                    "taker_buy_vol":  float(row[1]),
                    "taker_sell_vol": float(row[2]),
                }

        chunk_start = chunk_end
        time.sleep(0.2)

    # --- Inner join ---
    common_ts = set(candle_rows.keys()) & set(rubik_rows.keys())
    if not common_ts:
        log.warning(f"[{exchange}] OKX backfill: no overlapping timestamps, skipping")
        return

    records = []
    for ts_ms in sorted(common_ts):
        r = rubik_rows[ts_ms]
        c = candle_rows[ts_ms]
        records.append({
            "timestamp":     pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
            "exchange":      exchange,
            "open":          c["open"],
            "high":          c["high"],
            "low":           c["low"],
            "close":         c["close"],
            "volume":        r["taker_buy_vol"] + r["taker_sell_vol"],
            "taker_buy_vol": r["taker_buy_vol"],
        })

    result = pd.DataFrame(records)[
        ["timestamp", "exchange", "open", "high", "low", "close", "volume", "taker_buy_vol"]
    ]
    upsert_parquet(filepath, result)
    log.info(f"[{exchange}] OKX backfill done: {len(result)} candles")


def update_okx(
    exchange: str,
    inst_id: str,
    ccy: str,
    inst_type: str,
) -> None:
    """Fetch latest OKX candles and upsert to parquet (called every 60s)."""
    filepath = DATA_DIR / ("btc_spot_5m.parquet" if "spot" in exchange else "btc_futures_5m.parquet")
    df = fetch_okx_with_taker(exchange, inst_id, ccy, inst_type, limit=10)
    if not df.empty:
        n = upsert_parquet(filepath, df)
        log.info(f"[{exchange}] +{n} candles")


# ---------------------------------------------------------------------------
# TRADE BUFFER — thread-safe in-memory accumulator for polling exchanges
# ---------------------------------------------------------------------------

_trades_buffer: dict = {
    "bybit_spot":     pd.DataFrame(),
    "gate_spot":      pd.DataFrame(),
    "coinbase":       pd.DataFrame(),
    "kraken":         pd.DataFrame(),
    "bybit_futures":  pd.DataFrame(),
    "gate_futures":   pd.DataFrame(),
}

# One lock per exchange — protects concurrent access from polling threads vs flush thread
_buffer_locks: dict = {ex: threading.Lock() for ex in _trades_buffer}

# Exchange → market type, used by collect_cycle for flushing polling buffers
_POLLING_EXCHANGE_MARKET: dict = {
    "bybit_spot":    "spot",
    "gate_spot":     "spot",
    "coinbase":      "spot",
    "kraken":        "spot",
    "bybit_futures": "futures",
    "gate_futures":  "futures",
}


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
        log.info(f"[{exchange}] {market_type}: +{n} candles (polling)")


# ---------------------------------------------------------------------------
# POLLING RUNNERS — stateful REST fetch factories + PollingRunner class
# ---------------------------------------------------------------------------

class PollingRunner:
    """REST polling loop: calls fetch_fn every POLL_INTERVAL_S seconds,
    adds results to trade buffer. fetch_fn is stateful (closure with last_ts/cursor)."""

    POLL_INTERVAL_S = 10

    def __init__(self, exchange: str, fetch_fn: Callable[[], pd.DataFrame]) -> None:
        """
        Args:
            exchange: exchange key matching _trades_buffer
            fetch_fn: stateful callable — returns new trades since last call
        """
        self.exchange = exchange
        self.fetch_fn = fetch_fn

    def start(self) -> None:
        """Start polling in a daemon thread."""
        threading.Thread(
            target=self._run, daemon=True,
            name=f"poll-{self.exchange}",
        ).start()

    def _run(self) -> None:
        """Main poll loop."""
        while True:
            try:
                new_trades = self.fetch_fn()
                if not new_trades.empty:
                    update_buffer(self.exchange, new_trades)
            except Exception as e:
                log.warning(f"[{self.exchange}] poll error: {e}")
            time.sleep(self.POLL_INTERVAL_S)


def _make_bybit_fetcher(category: str) -> Callable[[], pd.DataFrame]:
    """
    Stateful fetch factory for Bybit spot (category='spot') or futures (category='linear').
    side == 'Buy' is taker buy (taker side, verified).
    """
    state: dict = {"last_ts_ms": now_ms()}

    def fetch() -> pd.DataFrame:
        url = "https://api.bybit.com/v5/market/recent-trade"
        params = {"category": category, "symbol": "BTCUSDT", "limit": 1000}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("retCode") != 0:
            log.warning(f"[bybit {category}] trades error: {data.get('retMsg')}")
            return pd.DataFrame()
        trades = data["result"]["list"]
        if not trades:
            return pd.DataFrame()
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["time"].astype(float), unit="ms", utc=True)
        df["price"]     = df["price"].astype(float)
        df["size"]      = df["size"].astype(float)
        df["is_buyer"]  = df["side"] == "Buy"
        df = df[df["timestamp"] > pd.Timestamp(state["last_ts_ms"], unit="ms", tz="UTC")]
        if df.empty:
            return pd.DataFrame()
        state["last_ts_ms"] = int(df["timestamp"].max().timestamp() * 1000)
        return df[["timestamp", "price", "size", "is_buyer"]]

    return fetch


def _make_gate_spot_fetcher() -> Callable[[], pd.DataFrame]:
    """
    Stateful fetch factory for Gate.io spot.
    REST side is TAKER side (same as WS): 'buy' = taker bought = is_buyer=True.
    Verified empirically 2026-04-08: sell% tracks price direction correctly.
    create_time_ms is float ms.
    """
    state: dict = {"last_unix_sec": int(datetime.now(timezone.utc).timestamp())}

    def fetch() -> pd.DataFrame:
        url = "https://api.gateio.ws/api/v4/spot/trades"
        params = {
            "currency_pair": "BTC_USDT",
            "limit": 1000,
            "from": state["last_unix_sec"],
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        trades = resp.json()
        if not trades:
            return pd.DataFrame()
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["create_time_ms"].astype(float), unit="ms", utc=True)
        df["price"]     = df["price"].astype(float)
        df["size"]      = df["amount"].astype(float)
        # taker side: 'buy' = taker bought
        df["is_buyer"]  = df["side"] == "buy"
        max_ts = df["timestamp"].max()
        state["last_unix_sec"] = int(max_ts.timestamp())
        return df[["timestamp", "price", "size", "is_buyer"]]

    return fetch


def _make_gate_futures_fetcher() -> Callable[[], pd.DataFrame]:
    """
    Stateful fetch factory for Gate.io USDT perpetual futures.
    size > 0 = buy, size < 0 = sell (no 'side' field).
    size in contracts, 1 contract = 1 USD → size_btc = abs(size) / price.
    create_time is float seconds.
    """
    state: dict = {"last_unix_sec": int(datetime.now(timezone.utc).timestamp())}

    def fetch() -> pd.DataFrame:
        url = "https://api.gateio.ws/api/v4/futures/usdt/trades"
        params = {
            "contract": "BTC_USDT",
            "limit": 1000,
            "from": state["last_unix_sec"],
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        trades = resp.json()
        if not trades:
            return pd.DataFrame()
        df = pd.DataFrame(trades)
        df["timestamp"] = pd.to_datetime(df["create_time"].astype(float), unit="s", utc=True)
        df["price"]     = df["price"].astype(float)
        df["size_raw"]  = df["size"].astype(float)
        df["is_buyer"]  = df["size_raw"] > 0
        df["size"]      = df["size_raw"].abs() / df["price"]
        max_ts = df["timestamp"].max()
        state["last_unix_sec"] = int(max_ts.timestamp())
        return df[["timestamp", "price", "size", "is_buyer"]]

    return fetch


def _make_kraken_fetcher() -> Callable[[], pd.DataFrame]:
    """
    Stateful fetch factory for Kraken.
    since = last nonce from previous response (not a timestamp).
    First poll: no since → last ~1000 trades.
    side == 'b' is taker buy.
    Rate limit: 15 req/min → polling at 10s = 6 req/min (safe).
    """
    state: dict = {"last_nonce": None}

    def fetch() -> pd.DataFrame:
        url = "https://api.kraken.com/0/public/Trades"
        params: dict = {"pair": "XBTUSD"}
        if state["last_nonce"] is not None:
            params["since"] = state["last_nonce"]
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            log.warning(f"[kraken] trades error: {data['error']}")
            return pd.DataFrame()
        result = data["result"]
        # update nonce for next call
        state["last_nonce"] = result.get("last")
        pair_key = next((k for k in result if k != "last"), None)
        if not pair_key:
            return pd.DataFrame()
        trades = result[pair_key]
        if not trades:
            return pd.DataFrame()
        df = pd.DataFrame(
            trades,
            columns=["price", "volume", "time", "side", "order_type", "misc", "trade_id"],
        )
        df["timestamp"] = pd.to_datetime(df["time"].astype(float), unit="s", utc=True)
        df["price"]     = df["price"].astype(float)
        df["size"]      = df["volume"].astype(float)
        df["is_buyer"]  = df["side"] == "b"
        return df[["timestamp", "price", "size", "is_buyer"]]

    return fetch


def _make_coinbase_fetcher() -> Callable[[], pd.DataFrame]:
    """
    Stateful fetch factory for Coinbase Exchange.
    Pagination: before=<trade_id> returns trades with id > before (newer trades).
    REST Exchange side: side == 'buy' = taker buy.
    trade_id is integer.
    """
    state: dict = {"before": None}

    def fetch() -> pd.DataFrame:
        url = "https://api.exchange.coinbase.com/products/BTC-USD/trades"
        params: dict = {"limit": 100}
        if state["before"] is not None:
            params["before"] = state["before"]
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        trades = resp.json()
        if not trades:
            return pd.DataFrame()
        df = pd.DataFrame(trades)
        df["timestamp"]  = pd.to_datetime(df["time"], utc=True)
        df["price"]      = df["price"].astype(float)
        df["size"]       = df["size"].astype(float)
        df["is_buyer"]   = df["side"] == "buy"
        df["trade_id"]   = df["trade_id"].astype(int)
        max_id = int(df["trade_id"].max())
        # On first poll just record the cursor; return empty to avoid double-counting
        if state["before"] is None:
            state["before"] = max_id
            return pd.DataFrame()
        state["before"] = max_id
        return df[["timestamp", "price", "size", "is_buyer"]]

    return fetch


def start_polling_runners() -> None:
    """Start REST polling runners for all trade-based exchanges."""
    runners: List[PollingRunner] = [
        PollingRunner("bybit_spot",    _make_bybit_fetcher("spot")),
        PollingRunner("bybit_futures", _make_bybit_fetcher("linear")),
        PollingRunner("gate_spot",     _make_gate_spot_fetcher()),
        PollingRunner("gate_futures",  _make_gate_futures_fetcher()),
        PollingRunner("kraken",        _make_kraken_fetcher()),
        PollingRunner("coinbase",      _make_coinbase_fetcher()),
    ]
    for runner in runners:
        runner.start()
        time.sleep(0.2)
    log.info(f"[polling] Started {len(runners)} REST polling runners")


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
    Binance klines and OKX candles are fetched via REST.
    Polling exchanges (bybit/gate/coinbase/kraken) feed through PollingRunner threads —
    this cycle only flushes their accumulated buffers.
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

    # 3. OKX spot (REST rubik + candles server-side aggregation)
    try:
        update_okx("okx_spot", "BTC-USDT", "BTC", "SPOT")
    except Exception as e:
        log.error(f"[okx_spot] update error: {e}")

    # 4. OKX futures (REST rubik + candles server-side aggregation)
    try:
        update_okx("okx_futures", "BTC-USDT-SWAP", "BTC", "CONTRACTS")
    except Exception as e:
        log.error(f"[okx_futures] update error: {e}")

    # 5. Flush polling trade buffers to parquet
    for exchange, market_type in _POLLING_EXCHANGE_MARKET.items():
        try:
            flush_buffer_to_parquet(exchange, market_type)
        except Exception as e:
            log.error(f"[{exchange}] flush error: {e}")

    # 6. OI snapshot
    update_oi_binance()


# ---------------------------------------------------------------------------
# BACKFILL
# ---------------------------------------------------------------------------

def run_backfill() -> None:
    """
    One-time backfill on startup.
    Native klines: full 5-day history (Binance spot + futures).
    OI: 5-day history via Binance 5m historical endpoint.
    OKX: 5-day history via server-side aggregation (rubik + candles).
    Polling exchanges: no backfill — they start collecting from now.
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

    # OKX server-side aggregation backfill
    try:
        backfill_okx("okx_spot", "BTC-USDT", "BTC", "SPOT")
    except Exception as e:
        log.error(f"Backfill [okx_spot] failed: {e}")
    time.sleep(0.5)

    try:
        backfill_okx("okx_futures", "BTC-USDT-SWAP", "BTC", "CONTRACTS")
    except Exception as e:
        log.error(f"Backfill [okx_futures] failed: {e}")

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
    "binance", "bybit_spot", "okx_spot",
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
    """Backfill historical data, start REST polling runners, then run the collection loop."""
    DATA_DIR.mkdir(exist_ok=True)
    run_backfill()
    start_polling_runners()

    alert_state = _load_alert_state()
    log.info(
        f"Collector running — update interval: {UPDATE_INTERVAL_S}s, "
        f"{len(_POLLING_EXCHANGE_MARKET)} polling streams active"
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
