#!/usr/bin/env python3
"""
backfill_signal_log.py — One-time backfill of signal_log.csv.

Run once on VPS from ~/DivScanner directory:
  docker exec divscanner-collector-1 python3 backfill_signal_log.py

What it does:
  1. Normalizes signal names: BUYERS/SELLERS → BUYING/SELLING
  2. Recovers btc_price from the misaligned price_atr_ratio column (old rows)
  3. Recomputes metrics for old rows using cvd_mode='candle' (original logic)
  4. Adds cvd_mode column: 'candle' for all existing rows, 'line' for future rows
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR        = Path("data")
SIGNAL_LOG_FILE = DATA_DIR / "signal_log.csv"

NAME_MAP = {
    "BUYERS EXHAUSTION":  "BUYING EXHAUSTION",
    "BUYERS ABSORPTION":  "BUYING ABSORPTION",
    "SELLERS EXHAUSTION": "SELLING EXHAUSTION",
    "SELLERS ABSORPTION": "SELLING ABSORPTION",
}

ALL_SPOT_EXCHANGES = [
    "binance", "mexc", "bybit_spot", "okx_spot",
    "gate_spot", "coinbase", "kraken",
]

TF_PRIORITY = {"1h": 4, "30m": 3, "15m": 2, "5m": 1}


def _top_tf(timeframes_str: str) -> str:
    tfs = [t.strip() for t in str(timeframes_str).split(",")]
    return max(tfs, key=lambda t: TF_PRIORITY.get(t, 0))


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _backfill_row(row: pd.Series, spot_5m: pd.DataFrame,
                  futures_5m: pd.DataFrame, oi_1m: pd.DataFrame):
    """
    Recompute quality metrics for one old signal row using cvd_mode='candle'.
    Returns dict of updated fields, or None if signal is not recoverable.
    """
    from analysis import (
        INTERVAL_MAP, DISPLAY_CANDLES,
        resample_klines, compute_cvd, trim_to_candles, reset_cvd_origin,
        get_price_df, detect_spot_signals,
    )

    signal_ts = pd.Timestamp(row["timestamp"])
    signal    = str(row["signal"])
    top_tf    = _top_tf(row["timeframes"])
    interval  = INTERVAL_MAP.get(top_tf, "5min")

    # --- Reconstruct price + CVD as seen at signal time ---
    spot_hist = spot_5m[spot_5m["timestamp"] <= signal_ts]
    if spot_hist.empty:
        return None

    spot_rs  = resample_klines(spot_hist, interval)
    price_df = trim_to_candles(get_price_df(spot_rs), DISPLAY_CANDLES)
    cvd_df   = reset_cvd_origin(
        trim_to_candles(compute_cvd(spot_rs, ALL_SPOT_EXCHANGES), DISPLAY_CANDLES)
    )

    if price_df.empty or cvd_df.empty:
        return None

    # --- Detect with original candle mode ---
    low_data, high_data = detect_spot_signals(price_df, cvd_df, cvd_mode="candle")

    # Select candidate matching signal direction
    is_sell   = "SELL" in signal.upper()
    candidate = low_data if is_sell else high_data

    if candidate is None:
        return None

    # Verify signal timestamp matches (must be exact same candle)
    if pd.Timestamp(candidate["timestamp"]) != signal_ts:
        return None

    updates = {
        "persistence":     candidate.get("persistence"),
        "price_atr_ratio": candidate.get("price_atr_ratio"),
        "cvd_sigma":       candidate.get("cvd_sigma"),
        "window_bars":     candidate.get("window_bars"),
    }

    pivot_from_ts = candidate.get("pivot_from_ts")

    # --- OI delta ---
    if pivot_from_ts is not None and not oi_1m.empty:
        try:
            oi_s = oi_1m.set_index("timestamp").sort_index()
            oi_a = float(oi_s["oi_value"].asof(pd.Timestamp(pivot_from_ts)))
            oi_b = float(oi_s["oi_value"].asof(pd.Timestamp(signal_ts)))
            if pd.notna(oi_a) and pd.notna(oi_b) and oi_a > 0:
                updates["oi_delta_pct"] = (oi_b - oi_a) / oi_a * 100
        except Exception as exc:
            log.debug(f"OI backfill failed: {exc}")

    # --- Futures CVD delta ---
    if pivot_from_ts is not None and not futures_5m.empty:
        try:
            fut_hist    = futures_5m[futures_5m["timestamp"] <= signal_ts]
            fut_rs      = resample_klines(fut_hist, interval)
            fut_exchs   = list(futures_5m["exchange"].unique())
            cvd_fut     = reset_cvd_origin(
                trim_to_candles(compute_cvd(fut_rs, fut_exchs), DISPLAY_CANDLES)
            )
            if not cvd_fut.empty:
                fs    = cvd_fut.set_index("timestamp").sort_index()
                fut_a = float(fs["cvd_close"].asof(pd.Timestamp(pivot_from_ts)))
                fut_b = float(fs["cvd_close"].asof(pd.Timestamp(signal_ts)))
                if pd.notna(fut_a) and pd.notna(fut_b):
                    updates["futures_cvd_delta"] = fut_b - fut_a
        except Exception as exc:
            log.debug(f"Futures CVD backfill failed: {exc}")

    return updates


def main() -> None:
    if not SIGNAL_LOG_FILE.exists():
        log.error("signal_log.csv not found")
        return

    df = pd.read_csv(SIGNAL_LOG_FILE)
    log.info(f"Loaded {len(df)} rows")

    # --- 1. Normalize signal names ---
    df["signal"] = df["signal"].replace(NAME_MAP)
    log.info(f"Signal names normalized")

    # --- 2. Identify old rows (cvd_sigma is NaN = old format, metrics misaligned) ---
    old_mask = df["cvd_sigma"].isna()
    log.info(f"Old rows needing backfill: {old_mask.sum()}")

    # --- 3. Recover btc_price; clear misaligned persistence / price_atr_ratio ---
    df.loc[old_mask, "btc_price"]      = df.loc[old_mask, "price_atr_ratio"]
    df.loc[old_mask, "persistence"]    = None
    df.loc[old_mask, "price_atr_ratio"] = None
    log.info(f"Recovered btc_price for {old_mask.sum()} rows, cleared misaligned values")

    # --- 4. Add cvd_mode column (all existing rows used candle detection) ---
    if "cvd_mode" not in df.columns:
        df["cvd_mode"] = "candle"
    else:
        df.loc[df["cvd_mode"].isna(), "cvd_mode"] = "candle"

    # --- 5. Backfill metrics ---
    spot_5m    = _load_parquet(DATA_DIR / "btc_spot_5m.parquet")
    futures_5m = _load_parquet(DATA_DIR / "btc_futures_5m.parquet")
    oi_1m      = _load_parquet(DATA_DIR / "btc_oi_1m.parquet")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    ok = fail = 0
    for idx in df[old_mask].index:
        row = df.loc[idx]
        try:
            updates = _backfill_row(row, spot_5m, futures_5m, oi_1m)
            if updates:
                for col, val in updates.items():
                    df.at[idx, col] = val
                ok += 1
                log.info(
                    f"[{idx:>2}] {row['signal']:<22} {str(row['timeframes']):<10} "
                    f"@ {str(row['timestamp'])[:16]} → ok  "
                    f"persist={updates.get('persistence', '?'):.3f}  "
                    f"atr={updates.get('price_atr_ratio', '?'):.3f}  "
                    f"σ={updates.get('cvd_sigma', '?'):.3f}"
                )
            else:
                fail += 1
                log.warning(f"[{idx:>2}] {row['signal']} @ {str(row['timestamp'])[:16]} → not recoverable")
        except Exception as exc:
            fail += 1
            log.warning(f"[{idx:>2}] {row['signal']} @ {str(row['timestamp'])[:16]} → error: {exc}")

    log.info(f"Backfill: {ok} recovered, {fail} unrecoverable")

    # --- 6. Save ---
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    df.to_csv(SIGNAL_LOG_FILE, index=False)
    log.info(f"Saved {len(df)} rows → {SIGNAL_LOG_FILE}")


if __name__ == "__main__":
    main()
