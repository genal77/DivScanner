#!/usr/bin/env python3
"""
backfill_signal_log.py — One-time backfill/repair of signal_log.csv.

Run inside the collector container:
  docker exec divscanner-collector-1 python3 backfill_signal_log.py

Steps performed:
  1. Normalize signal names (BUYERS/SELLERS → BUYING/SELLING)
  2. Split multi-TF rows into one row per TF (each with its own metrics)
  3. Recover btc_price from the misaligned price_atr_ratio column (old rows)
  4. Set cvd_mode='candle' for all existing rows
  5. Recompute per-TF metrics (persistence, price_atr_ratio, cvd_sigma,
     window_bars, oi_delta_btc, futures_cvd_delta) from parquet history
  6. Rename oi_delta_pct → oi_delta_btc (absolute BTC, not %)
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

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


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _backfill_row(row: pd.Series, tf: str,
                  spot_5m: pd.DataFrame,
                  futures_5m: pd.DataFrame,
                  oi_1m: pd.DataFrame) -> Optional[dict]:
    """
    Recompute all quality metrics for one signal at a specific TF.
    Uses cvd_mode from the row itself to match the original detection logic.
    Returns dict of updated fields, or None if unrecoverable.
    """
    from analysis import (
        INTERVAL_MAP, DISPLAY_CANDLES,
        resample_klines, compute_cvd, trim_to_candles, reset_cvd_origin,
        get_price_df, detect_spot_signals,
    )

    signal_ts = pd.Timestamp(row["timestamp"])
    signal    = str(row["signal"])
    interval  = INTERVAL_MAP.get(tf, "5min")
    cvd_mode  = str(row.get("cvd_mode", "candle")) if pd.notna(row.get("cvd_mode")) else "candle"

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

    low_data, high_data = detect_spot_signals(price_df, cvd_df, cvd_mode=cvd_mode)

    is_sell   = "SELL" in signal.upper()
    candidate = low_data if is_sell else high_data

    if candidate is None:
        return None
    if pd.Timestamp(candidate["timestamp"]) != signal_ts:
        return None

    pivot_from_ts = candidate.get("pivot_from_ts")

    updates = {
        "pivot_from_ts":   pd.Timestamp(pivot_from_ts).isoformat() if pivot_from_ts is not None else None,
        "price_from":      candidate.get("price_from"),
        "price_to":        candidate.get("price_to"),
        "cvd_from":        candidate.get("cvd_from"),
        "cvd_to":          candidate.get("cvd_to"),
        "price_move_pct":  candidate.get("price_move_pct"),
        "cvd_move_pct":    candidate.get("cvd_move_pct"),
        "persistence":     candidate.get("persistence"),
        "price_atr_ratio": candidate.get("price_atr_ratio"),
        "cvd_sigma":       candidate.get("cvd_sigma"),
        "window_bars":     candidate.get("window_bars"),
    }

    # OI delta (absolute BTC)
    if pivot_from_ts is not None and not oi_1m.empty:
        try:
            oi_s = oi_1m.set_index("timestamp").sort_index()
            oi_a = float(oi_s["oi_value"].asof(pd.Timestamp(pivot_from_ts)))
            oi_b = float(oi_s["oi_value"].asof(pd.Timestamp(signal_ts)))
            if pd.notna(oi_a) and pd.notna(oi_b):
                updates["oi_delta_btc"] = oi_b - oi_a
        except Exception as exc:
            log.debug(f"OI backfill failed: {exc}")

    # Futures CVD delta
    if pivot_from_ts is not None and not futures_5m.empty:
        try:
            fut_hist  = futures_5m[futures_5m["timestamp"] <= signal_ts]
            fut_rs    = resample_klines(fut_hist, interval)
            fut_exchs = list(futures_5m["exchange"].unique())
            cvd_fut   = reset_cvd_origin(
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

    # 1. Normalize signal names
    df["signal"] = df["signal"].replace(NAME_MAP)
    log.info("Signal names normalized")

    # 2. Add pivot_from_ts column if missing
    if "pivot_from_ts" not in df.columns:
        df["pivot_from_ts"] = None

    # 3. Rename oi_delta_pct → oi_delta_btc if old column present
    if "oi_delta_pct" in df.columns and "oi_delta_btc" not in df.columns:
        df.rename(columns={"oi_delta_pct": "oi_delta_btc"}, inplace=True)
        df["oi_delta_btc"] = None  # will be recomputed from parquet
        log.info("Renamed oi_delta_pct → oi_delta_btc (values cleared for recompute)")
    elif "oi_delta_btc" not in df.columns:
        df["oi_delta_btc"] = None

    # 3. Add cvd_mode column (all existing = candle)
    if "cvd_mode" not in df.columns:
        df["cvd_mode"] = "candle"
    else:
        df.loc[df["cvd_mode"].isna(), "cvd_mode"] = "candle"

    # 4. Split multi-TF rows into one row per TF
    expanded = []
    for _, row in df.iterrows():
        tfs = [t.strip() for t in str(row["timeframes"]).split(",")]
        if len(tfs) == 1:
            expanded.append(row.to_dict())
        else:
            for tf in tfs:
                new_row = row.to_dict()
                new_row["timeframes"] = tf
                # Mark quality metrics as needing recompute (may be from wrong TF)
                new_row["pivot_from_ts"]   = None
                new_row["persistence"]     = None
                new_row["price_atr_ratio"] = None
                new_row["cvd_sigma"]       = None
                new_row["window_bars"]     = None
                new_row["oi_delta_btc"]    = None
                new_row["futures_cvd_delta"] = None
                expanded.append(new_row)

    orig_count = len(df)
    df = pd.DataFrame(expanded)
    log.info(f"After split: {orig_count} → {len(df)} rows")

    # 5. Recover btc_price from misaligned price_atr_ratio for rows where
    #    btc_price is missing AND price_atr_ratio looks like a BTC price (> 1000)
    old_mask = df["cvd_sigma"].isna()
    btc_price_missing = old_mask & df["btc_price"].isna() & (pd.to_numeric(df["price_atr_ratio"], errors="coerce") > 1000)
    df.loc[btc_price_missing, "btc_price"]       = df.loc[btc_price_missing, "price_atr_ratio"]
    df.loc[btc_price_missing, "price_atr_ratio"] = None
    if btc_price_missing.sum():
        log.info(f"Recovered btc_price for {btc_price_missing.sum()} rows")

    # 6. Recompute metrics for all rows needing it (cvd_sigma is NaN)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    spot_5m    = _load_parquet(DATA_DIR / "btc_spot_5m.parquet")
    futures_5m = _load_parquet(DATA_DIR / "btc_futures_5m.parquet")
    oi_1m      = _load_parquet(DATA_DIR / "btc_oi_1m.parquet")

    # Full recompute: rows missing cvd_sigma
    # OI-only recompute: rows that have cvd_sigma but lost oi_delta_btc during rename
    needs_backfill  = df["cvd_sigma"].isna()
    needs_oi_only   = ~df["cvd_sigma"].isna() & df["oi_delta_btc"].isna()
    log.info(f"Rows needing full recompute: {needs_backfill.sum()}, OI-only: {needs_oi_only.sum()}")

    ok = fail = 0
    for idx in df[needs_backfill].index:
        row = df.loc[idx]
        tf  = str(row["timeframes"]).strip()
        try:
            updates = _backfill_row(row, tf, spot_5m, futures_5m, oi_1m)
            if updates:
                for col, val in updates.items():
                    df.at[idx, col] = val
                ok += 1
                log.info(
                    f"[{idx:>2}] {str(row['signal']):<22} {tf:<5} "
                    f"@ {str(row['timestamp'])[:16]} → ok  "
                    f"persist={updates.get('persistence', float('nan')):.3f}  "
                    f"atr={updates.get('price_atr_ratio', float('nan')):.3f}  "
                    f"σ={updates.get('cvd_sigma', float('nan')):.3f}"
                )
            else:
                fail += 1
                log.warning(f"[{idx:>2}] {str(row['signal']):<22} {tf:<5} @ {str(row['timestamp'])[:16]} → not recoverable")
        except Exception as exc:
            fail += 1
            log.warning(f"[{idx:>2}] error: {exc}")

    log.info(f"Backfill: {ok} recovered, {fail} unrecoverable")

    # 6b. OI-only pass: rows that have cvd_sigma but lost oi_delta_btc
    oi_ok = oi_fail = 0
    for idx in df[needs_oi_only].index:
        row = df.loc[idx]
        tf  = str(row["timeframes"]).strip()
        try:
            updates = _backfill_row(row, tf, spot_5m, futures_5m, oi_1m)
            if updates and updates.get("oi_delta_btc") is not None:
                df.at[idx, "oi_delta_btc"]      = updates["oi_delta_btc"]
                df.at[idx, "futures_cvd_delta"]  = updates.get("futures_cvd_delta")
                df.at[idx, "pivot_from_ts"]      = updates.get("pivot_from_ts")
                oi_ok += 1
                log.info(f"[{idx:>2}] OI filled: {str(row['signal']):<22} {tf:<5} → oi_delta_btc={updates['oi_delta_btc']:+.1f}")
            else:
                oi_fail += 1
                log.warning(f"[{idx:>2}] OI not recoverable: {str(row['signal']):<22} {tf:<5}")
        except Exception as exc:
            oi_fail += 1
            log.warning(f"[{idx:>2}] OI error: {exc}")

    log.info(f"OI pass: {oi_ok} filled, {oi_fail} unrecoverable")

    # 7. Save
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    df.to_csv(SIGNAL_LOG_FILE, index=False)
    log.info(f"Saved {len(df)} rows → {SIGNAL_LOG_FILE}")


if __name__ == "__main__":
    main()
