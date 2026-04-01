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
  Trades aggregation:        bybit, okx, gate, coinbase, kraken

Futures exchanges:
  Native taker buy (klines): binance_futures
  Trades aggregation:        bybit_futures, okx_futures, gate_futures, mexc_futures

OI: binance_futures (polled every 60s, backfilled via 5m historical endpoint)
"""

import requests
import pandas as pd
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
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
    """Save DataFrame to parquet, pruning data older than HISTORY_DAYS."""
    if df.empty:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)
    if "timestamp" in df.columns:
        df = df[df["timestamp"] >= cutoff]
    df.to_parquet(filepath, index=False)


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
# SPOT — NATIVE TAKER BUY (BINANCE, MEXC)
# Both use Binance-compatible klines endpoint: field [9] = taker_buy_vol
# ---------------------------------------------------------------------------

SPOT_NATIVE_ENDPOINTS = {
    "binance": "https://api.binance.com/api/v3/klines",
}


def fetch_spot_klines_native(exchange: str, start_ms: int) -> pd.DataFrame:
    """Fetch spot klines with native taker buy volume (Binance/MEXC)."""
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
# TRADES FETCHERS — raw trades for exchanges without native taker buy
# Each returns a DataFrame with columns: timestamp, price, size, is_buyer
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
    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["create_time_ms"] if "create_time_ms" in df.columns else df["create_time"].astype(float) * 1000),
        unit="ms", utc=True,
    )
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


# Kraken tracks its own `since` cursor to avoid missing trades between polls
_kraken_since: Optional[str] = None


def fetch_trades_kraken() -> pd.DataFrame:
    """
    Fetch BTC/USD trades from Kraken.
    Uses `since` cursor to fetch only new trades on each call.
    """
    global _kraken_since
    url = "https://api.kraken.com/0/public/Trades"
    params = {"pair": "XBTUSD"}
    if _kraken_since:
        params["since"] = _kraken_since

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if data.get("error"):
        log.warning(f"Kraken trades error: {data['error']}")
        return pd.DataFrame()

    result = data["result"]
    _kraken_since = result.get("last")  # update cursor for next call

    # Kraken returns trades under the pair key (XXBTZUSD or XBTUSD)
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


# ---------------------------------------------------------------------------
# TRADES BUFFER — in-memory accumulator for trades-based exchanges
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


def update_buffer(exchange: str, new_trades: pd.DataFrame) -> None:
    """Add new trades to buffer, keep only last TRADES_BUFFER_MINUTES minutes."""
    if new_trades.empty:
        return
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=TRADES_BUFFER_MINUTES)
    combined = pd.concat([_trades_buffer[exchange], new_trades], ignore_index=True)
    combined = combined[combined["timestamp"] > cutoff].drop_duplicates()
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
    Only flushes candles that are fully closed (not the currently forming one).
    """
    buf = _trades_buffer[exchange]
    if buf.empty:
        return

    now = datetime.now(timezone.utc)
    # Round down to start of current 5m candle
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
        log.info(f"[{exchange}] {market_type}: +{n} candles (trades)")


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

# Maps exchange name → (fetcher_function, market_type_string)
TRADES_EXCHANGES = {
    "mexc_spot":     (fetch_trades_mexc_spot,                                 "spot"),
    "bybit_spot":    (lambda: fetch_trades_bybit(category="spot"),           "spot"),
    "okx_spot":      (lambda: fetch_trades_okx(inst_id="BTC-USDT"),          "spot"),
    "gate_spot":     (fetch_trades_gate_spot,                                 "spot"),
    "coinbase":      (fetch_trades_coinbase,                                  "spot"),
    "kraken":        (fetch_trades_kraken,                                    "spot"),
    "bybit_futures": (lambda: fetch_trades_bybit(category="linear"),          "futures"),
    "okx_futures":   (lambda: fetch_trades_okx(inst_id="BTC-USDT-SWAP"),     "futures"),
    "gate_futures":  (fetch_trades_gate_futures,                              "futures"),
    "mexc_futures":  (fetch_trades_mexc_futures,                              "futures"),
}


def collect_cycle() -> None:
    """One full data collection cycle (runs every UPDATE_INTERVAL_S seconds)."""

    # 1. Native klines (spot)
    for exchange in ["binance"]:
        try:
            update_spot_native(exchange)
        except Exception as e:
            log.error(f"[{exchange}] spot klines error: {e}")

    # 2. Native klines (futures)
    try:
        update_futures_binance()
    except Exception as e:
        log.error(f"[binance_futures] klines error: {e}")

    # 3. Trades-based exchanges (spot + futures)
    for exchange, (fetcher, market_type) in TRADES_EXCHANGES.items():
        try:
            new_trades = fetcher()
            update_buffer(exchange, new_trades)
            flush_buffer_to_parquet(exchange, market_type)
        except Exception as e:
            log.error(f"[{exchange}] trades error: {e}")

    # 4. OI snapshot
    update_oi_binance()


# ---------------------------------------------------------------------------
# BACKFILL
# ---------------------------------------------------------------------------

def run_backfill() -> None:
    """
    One-time backfill on startup.
    Native klines: full 5-day history.
    Trades-based: only recent trades available (last ~1000 per exchange).
    OI: 5-day history via Binance 5m historical endpoint.
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

    # Initial trades fetch for trades-based exchanges
    # (no historical data available — starts accumulating from now)
    log.info("Fetching initial trades snapshot for trades-based exchanges...")
    for exchange, (fetcher, market_type) in TRADES_EXCHANGES.items():
        try:
            new_trades = fetcher()
            update_buffer(exchange, new_trades)
            log.info(f"[{exchange}] initial trades: {len(new_trades)} rows")
        except Exception as e:
            log.error(f"[{exchange}] initial trades failed: {e}")
        time.sleep(0.3)

    log.info("=" * 50)
    log.info("Backfill complete.")
    log.info("=" * 50)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def start() -> None:
    """Backfill historical data, then run the collection loop indefinitely."""
    DATA_DIR.mkdir(exist_ok=True)
    run_backfill()

    log.info(f"Collector running — update interval: {UPDATE_INTERVAL_S}s")
    while True:
        try:
            collect_cycle()
        except Exception as e:
            log.error(f"Collection cycle failed: {e}")
        time.sleep(UPDATE_INTERVAL_S)


if __name__ == "__main__":
    start()
