"""
outcome_tracker.py — Post-signal price outcome tracker for backtesting.

For each divergence signal in signal_log.csv, computes and saves:
  - Entry price, SL levels (tight/normal/wide via ATR), TP levels (1R/2R/3R)
  - Time-to-hit for each SL and TP level independently (None = not reached in horizon)
  - chg_pct / MFE / MAE at fixed time horizons (varies by signal TF)

Column structure is fixed and complete — all horizon columns always present,
NaN for horizons not applicable to a given TF. This prevents CSV misalignment
when rows with different TFs are appended over time.

Writes results to data/signal_outcomes.csv.
Called once per collector cycle — skips signals with insufficient parquet data.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

DATA_DIR        = Path("data")
SIGNAL_LOG_FILE = DATA_DIR / "signal_log.csv"
OUTCOMES_FILE   = DATA_DIR / "signal_outcomes.csv"

# Measurement horizons in hours, keyed by signal TF.
TF_HORIZONS: Dict[str, List[float]] = {
    "5m":  [0.25, 0.5,  1.0,  2.0,  4.0],
    "15m": [0.5,  1.0,  2.0,  4.0,  8.0, 12.0],
    "30m": [1.0,  2.0,  4.0,  8.0, 12.0, 24.0],
    "1h":  [4.0,  8.0, 12.0, 24.0, 48.0],
}
TF_PRIORITY = {"1h": 4, "30m": 3, "15m": 2, "5m": 1}

# SL multiples of ATR applied below/above the pivot extreme
SL_CONFIGS = {"sl_tight": 0.0, "sl_normal": 0.75, "sl_wide": 1.5}
# TP multiples of (entry - sl_normal) risk
TP_CONFIGS = {"tp_1r": 1.0, "tp_2r": 2.0, "tp_3r": 3.0}
ATR_BARS   = 14

# ---------------------------------------------------------------------------
# FIXED COLUMN SCHEMA — written once, never changes across appends
# ---------------------------------------------------------------------------

# All possible horizon labels across all TFs, in chronological order
_ALL_HORIZON_LABELS = ["15m", "30m", "1h", "2h", "4h", "8h", "12h", "24h", "48h"]

_FIXED_COLS = [
    "sent_at", "timestamp", "signal", "timeframes", "direction",
    "entry", "pivot", "atr", "risk",
    "sl_tight", "sl_normal", "sl_wide",
    "tp_1r", "tp_2r", "tp_3r",
    "sl_tight_min", "sl_normal_min", "sl_wide_min",
    "tp_1r_min", "tp_2r_min", "tp_3r_min",
]

_HORIZON_COLS = []
for _lbl in _ALL_HORIZON_LABELS:
    _HORIZON_COLS += [f"chg_pct_{_lbl}", f"mfe_{_lbl}", f"mae_{_lbl}"]

ALL_COLS = _FIXED_COLS + _HORIZON_COLS


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _horizon_label(h: float) -> str:
    """0.25 → '15m', 1.0 → '1h', 24.0 → '24h'."""
    if h < 1.0:
        return f"{int(h * 60)}m"
    return f"{int(h)}h"


def _top_tf(timeframes_str: str) -> str:
    """Return highest-priority TF from comma-separated string, e.g. '30m,15m' → '30m'."""
    tfs = [t.strip() for t in timeframes_str.split(",")]
    return max(tfs, key=lambda t: TF_PRIORITY.get(t, 0))


def _get_horizons(timeframes_str: str) -> List[float]:
    tf = _top_tf(timeframes_str)
    return TF_HORIZONS.get(tf, TF_HORIZONS["5m"])


def _resample_to_tf(binance_5m: pd.DataFrame, pandas_interval: str) -> pd.DataFrame:
    """Resample Binance 5m OHLC to a higher interval."""
    if binance_5m.empty or pandas_interval == "5min":
        return binance_5m.copy()
    g = binance_5m.set_index("timestamp").sort_index()
    out = pd.DataFrame({
        "open":  g["open"].resample(pandas_interval).first(),
        "high":  g["high"].resample(pandas_interval).max(),
        "low":   g["low"].resample(pandas_interval).min(),
        "close": g["close"].resample(pandas_interval).last(),
    }).dropna(subset=["open"]).reset_index()
    return out


def _compute_atr(ohlc: pd.DataFrame, n: int = ATR_BARS) -> float:
    """Mean of (high - low) over the last n candles. Returns 0 if data is thin."""
    recent = ohlc.tail(n)
    if len(recent) < 2:
        return 0.0
    return float((recent["high"] - recent["low"]).mean())


# ---------------------------------------------------------------------------
# CORE OUTCOME CALCULATION
# ---------------------------------------------------------------------------

def _compute_outcome(row: pd.Series, spot_5m: pd.DataFrame) -> Optional[Dict]:
    """
    Compute all outcome metrics for a single signal row.

    Returns None if parquet data does not yet cover the max horizon —
    the signal will be retried on the next collector cycle.

    Uses 5m candles for MFE/MAE and hit-time detection (highest accuracy).
    Uses resampled TF candles for ATR (matches the signal's own timeframe).
    """
    from analysis import INTERVAL_MAP

    sent_at         = pd.Timestamp(row["sent_at"])
    signal          = str(row["signal"])
    tfs_str         = str(row["timeframes"])
    entry           = float(row["btc_price"])
    pivot           = float(row["price_to"])   # pivot extreme: low for bullish, high for bearish

    # "SELLING *" → sellers losing → bullish
    is_bullish      = "SELL" in signal.upper()
    top_tf_str      = _top_tf(tfs_str)
    pandas_interval = INTERVAL_MAP.get(top_tf_str, "5min")

    horizons      = _get_horizons(tfs_str)
    max_h         = max(horizons)
    max_ts_needed = sent_at + pd.Timedelta(hours=max_h)

    # --- Post-signal 5m candles (Binance only) ---
    binance_5m = spot_5m[spot_5m["exchange"] == "binance"].copy()
    post_5m    = binance_5m[binance_5m["timestamp"] > sent_at].sort_values("timestamp").reset_index(drop=True)

    if post_5m.empty or post_5m["timestamp"].max() < max_ts_needed:
        return None  # not enough data yet — retry later

    # --- ATR from pre-signal candles at signal TF ---
    pre_5m = binance_5m[binance_5m["timestamp"] <= sent_at]
    pre_tf = _resample_to_tf(pre_5m, pandas_interval)
    atr    = _compute_atr(pre_tf)

    # --- SL / TP levels ---
    if is_bullish:
        sl   = {k: pivot - mult * atr for k, mult in SL_CONFIGS.items()}
        risk = entry - sl["sl_normal"]
        tp   = {k: entry + mult * risk for k, mult in TP_CONFIGS.items()}
    else:
        sl   = {k: pivot + mult * atr for k, mult in SL_CONFIGS.items()}
        risk = sl["sl_normal"] - entry
        tp   = {k: entry - mult * risk for k, mult in TP_CONFIGS.items()}

    risk = max(risk, 0.0)

    # --- Build base outcome dict (fixed columns only, all present) ---
    outcome: Dict = {
        "sent_at":        row["sent_at"],
        "timestamp":      row["timestamp"],
        "signal":         signal,
        "timeframes":     tfs_str,
        "direction":      "LONG" if is_bullish else "SHORT",
        "entry":          round(entry, 2),
        "pivot":          round(pivot, 2),
        "atr":            round(atr, 2),
        "risk":           round(risk, 2),
        "sl_tight":       round(sl["sl_tight"],  2),
        "sl_normal":      round(sl["sl_normal"], 2),
        "sl_wide":        round(sl["sl_wide"],   2),
        "tp_1r":          round(tp["tp_1r"], 2),
        "tp_2r":          round(tp["tp_2r"], 2),
        "tp_3r":          round(tp["tp_3r"], 2),
        "sl_tight_min":   None,
        "sl_normal_min":  None,
        "sl_wide_min":    None,
        "tp_1r_min":      None,
        "tp_2r_min":      None,
        "tp_3r_min":      None,
    }

    # --- Horizon snapshots ---
    elapsed_h = (post_5m["timestamp"] - sent_at).dt.total_seconds() / 3600

    for h in horizons:
        label  = _horizon_label(h)
        in_win = post_5m[elapsed_h <= h]
        if in_win.empty:
            continue

        last_close = float(in_win["close"].iloc[-1])
        change_pct = (last_close - entry) / entry * 100
        if not is_bullish:
            change_pct = -change_pct  # positive = move in signal direction

        if is_bullish:
            mfe = float((in_win["high"] - entry).max())
            mae = float((entry - in_win["low"]).max())
        else:
            mfe = float((entry - in_win["low"]).max())
            mae = float((in_win["high"] - entry).max())

        outcome[f"chg_pct_{label}"] = round(change_pct, 4)
        outcome[f"mfe_{label}"]     = round(mfe, 2)
        outcome[f"mae_{label}"]     = round(mae, 2)

    # --- Independent hit-time detection for each SL and TP level ---
    # Each level is tracked separately — hitting sl_tight does NOT stop
    # tracking sl_normal or any TP. This allows comparing outcomes across
    # all SL/TP combinations in post-analysis.

    sl_check = [
        ("sl_tight_min",  sl["sl_tight"]),
        ("sl_normal_min", sl["sl_normal"]),
        ("sl_wide_min",   sl["sl_wide"]),
    ]
    tp_check = [
        ("tp_1r_min", tp["tp_1r"]),
        ("tp_2r_min", tp["tp_2r"]),
        ("tp_3r_min", tp["tp_3r"]),
    ]
    max_min = max_h * 60

    for _, candle in post_5m.iterrows():
        elapsed_min = (candle["timestamp"] - sent_at).total_seconds() / 60
        if elapsed_min > max_min:
            break

        for col, price in sl_check:
            if outcome[col] is None:
                if is_bullish and candle["low"] <= price:
                    outcome[col] = round(elapsed_min, 1)
                elif not is_bullish and candle["high"] >= price:
                    outcome[col] = round(elapsed_min, 1)

        for col, price in tp_check:
            if outcome[col] is None:
                if is_bullish and candle["high"] >= price:
                    outcome[col] = round(elapsed_min, 1)
                elif not is_bullish and candle["low"] <= price:
                    outcome[col] = round(elapsed_min, 1)

        # Early exit once all levels are hit
        if all(outcome[c] is not None for c in ["sl_tight_min", "sl_normal_min", "sl_wide_min",
                                                  "tp_1r_min", "tp_2r_min", "tp_3r_min"]):
            break

    return outcome


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def check_outcomes() -> None:
    """
    Load signals, compute outcomes for any that aren't yet resolved,
    and append results to signal_outcomes.csv.

    Safe to call repeatedly — already-computed outcomes are never rewritten.
    Unique key: (sent_at, timeframes) — handles multi-TF signals logged as
    separate rows with the same sent_at.
    """
    if not SIGNAL_LOG_FILE.exists():
        return

    signals_df = pd.read_csv(SIGNAL_LOG_FILE)
    if signals_df.empty or "btc_price" not in signals_df.columns:
        return

    # Build set of already-resolved (sent_at, timeframes) pairs
    if OUTCOMES_FILE.exists():
        outcomes_df = pd.read_csv(OUTCOMES_FILE)
        done = set(zip(outcomes_df["sent_at"].astype(str), outcomes_df["timeframes"].astype(str)))
    else:
        done = set()

    pending = signals_df[
        ~signals_df.apply(lambda r: (str(r["sent_at"]), str(r["timeframes"])) in done, axis=1)
    ]
    if pending.empty:
        return

    # Load parquet once for all pending signals
    spot_path = DATA_DIR / "btc_spot_5m.parquet"
    if not spot_path.exists():
        return
    spot_5m = pd.read_parquet(spot_path)
    if spot_5m.empty:
        return
    spot_5m["timestamp"] = pd.to_datetime(spot_5m["timestamp"], utc=True)

    new_rows = []
    for _, row in pending.iterrows():
        try:
            outcome = _compute_outcome(row, spot_5m)
            if outcome is not None:
                new_rows.append(outcome)
                log.info(
                    f"[outcomes] {row['signal']} {row['timeframes']} "
                    f"@ {row['sent_at']} → sl_normal_min={outcome['sl_normal_min']} "
                    f"tp_1r_min={outcome['tp_1r_min']}"
                )
        except Exception as exc:
            log.warning(f"[outcomes] failed for {row.get('sent_at')}: {exc}")

    if not new_rows:
        return

    # Enforce fixed column order — fills missing horizon cols with NaN
    new_df    = pd.DataFrame(new_rows).reindex(columns=ALL_COLS)
    write_hdr = not OUTCOMES_FILE.exists()
    new_df.to_csv(OUTCOMES_FILE, mode="a", header=write_hdr, index=False)
    log.info(f"[outcomes] appended {len(new_rows)} rows to signal_outcomes.csv")
