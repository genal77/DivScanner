"""
analysis.py — Shared data processing, divergence detection, and figure building.

Imported by both app.py (display) and collector.py (alerts).
No Dash dependencies — only pandas and plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

_WARSAW = ZoneInfo("Europe/Warsaw")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PIVOT_WINDOW    = 5    # candles on each side required to confirm a pivot
DISPLAY_CANDLES = 200  # candles shown in main chart
ALERT_CANDLES   = 60   # candles shown in Telegram alert screenshot

INTERVAL_MAP = {
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
}
INTERVAL_SECONDS = {
    "5m":  5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h":  60 * 60,
}
_NICE_TICK_MS = [h * 3_600_000 for h in (1, 2, 3, 4, 6, 8, 12, 24, 48)]

SPOT_EXCHANGES = {
    "binance":    "Binance",
    "mexc":       "MEXC",
    "bybit_spot": "Bybit",
    "okx_spot":   "OKX",
    "gate_spot":  "Gate",
    "coinbase":   "Coinbase",
    "kraken":     "Kraken",
}

FUTURES_EXCHANGES = {
    "binance_futures": "Binance",
    "bybit_futures":   "Bybit",
    "okx_futures":     "OKX",
    "gate_futures":    "Gate",
    "mexc_futures":    "MEXC",
}

SIGNAL_EMOJI = {
    "SELLING EXHAUSTION": "🟢",
    "SELLING ABSORPTION": "🔵",
    "BUYING EXHAUSTION":  "🔴",
    "BUYING ABSORPTION":  "🟠",
}

CANDLE_UP   = "#4caf50"
CANDLE_DOWN = "#ef5350"

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_parquet(filepath) -> pd.DataFrame:
    """Load parquet file. Returns empty DataFrame if file not found."""
    from pathlib import Path
    filepath = Path(filepath)
    if not filepath.exists():
        return pd.DataFrame()
    df = pd.read_parquet(filepath)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

# ---------------------------------------------------------------------------
# DATA PROCESSING
# ---------------------------------------------------------------------------

def resample_klines(df: pd.DataFrame, pandas_interval: str) -> pd.DataFrame:
    """
    Resample 5m klines to a higher interval, per exchange.

    OHLC rules: open=first, high=max, low=min, close=last
    Volume rules: sum (additive across the period)
    """
    if df.empty:
        return df
    if pandas_interval == "5min":
        return df

    results = []
    for exchange, group in df.groupby("exchange"):
        g = group.set_index("timestamp").sort_index()
        resampled = pd.DataFrame({
            "open":          g["open"].resample(pandas_interval).first(),
            "high":          g["high"].resample(pandas_interval).max(),
            "low":           g["low"].resample(pandas_interval).min(),
            "close":         g["close"].resample(pandas_interval).last(),
            "volume":        g["volume"].resample(pandas_interval).sum(),
            "taker_buy_vol": g["taker_buy_vol"].resample(pandas_interval).sum(),
        }).dropna(subset=["open"]).reset_index()
        resampled["exchange"] = exchange
        results.append(resampled)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_cvd(klines_df: pd.DataFrame, selected_exchanges: List[str]) -> pd.DataFrame:
    """
    Compute aggregated CVD OHLC candles from klines of selected exchanges.

    Logic:
      1. Sum volume + taker_buy_vol per timestamp across all selected exchanges
      2. delta per candle = taker_buy_vol - taker_sell_vol
      3. CVD_Close = cumulative sum of delta
      4. CVD_Open = CVD_Close shifted by 1
      5. CVD_High/Low = max/min of Open and Close
    """
    if klines_df.empty or not selected_exchanges:
        return pd.DataFrame()

    df = klines_df[klines_df["exchange"].isin(selected_exchanges)]
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby("timestamp", as_index=False)
        .agg(volume=("volume", "sum"), taker_buy_vol=("taker_buy_vol", "sum"))
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    agg["delta"]     = agg["taker_buy_vol"] - (agg["volume"] - agg["taker_buy_vol"])
    agg["cvd_close"] = agg["delta"].cumsum()
    agg["cvd_open"]  = agg["cvd_close"].shift(1).fillna(agg["cvd_close"].iloc[0])
    agg["cvd_high"]  = agg[["cvd_open", "cvd_close"]].max(axis=1)
    agg["cvd_low"]   = agg[["cvd_open", "cvd_close"]].min(axis=1)

    return agg


def reset_cvd_origin(cvd_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize CVD to start from 0 at the first visible candle.

    Subtracts the cvd_open of the first row from all CVD columns so the
    curve always begins at 0 regardless of historical accumulation.
    Call this after trim_to_candles, not before.
    """
    if cvd_df.empty:
        return cvd_df
    df = cvd_df.copy()
    origin = df["cvd_open"].iloc[0]
    df["cvd_open"]  -= origin
    df["cvd_close"] -= origin
    df["cvd_high"]  -= origin
    df["cvd_low"]   -= origin
    return df


def compute_oi_ohlc(oi_df: pd.DataFrame, pandas_interval: str) -> pd.DataFrame:
    """Resample 1m OI snapshots into OHLC candles for the target interval."""
    if oi_df.empty:
        return pd.DataFrame()

    oi = oi_df.set_index("timestamp").sort_index()
    # Forward-fill missing 1m snapshots before resampling.
    # The collector polls OI every ~60s but cycle overhead causes ~1 missed snapshot
    # per 9 cycles. OI changes smoothly so filling with the previous value is correct.
    oi = oi.resample("1min").last().ffill()
    resampled = pd.DataFrame({
        "open":  oi["oi_value"].resample(pandas_interval).first(),
        "high":  oi["oi_value"].resample(pandas_interval).max(),
        "low":   oi["oi_value"].resample(pandas_interval).min(),
        "close": oi["oi_value"].resample(pandas_interval).last(),
    }).dropna(subset=["open"]).reset_index()

    return resampled


def trim_to_candles(df: pd.DataFrame, n: int, ts_col: str = "timestamp") -> pd.DataFrame:
    """Return only the last n candles, sorted by timestamp."""
    if df.empty:
        return df
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df.iloc[-n:].reset_index(drop=True)


def get_price_df(spot_df: pd.DataFrame) -> pd.DataFrame:
    """Extract Binance spot klines as the reference price series."""
    if spot_df.empty:
        return pd.DataFrame()
    df = spot_df[spot_df["exchange"] == "binance"]
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# DIVERGENCE DETECTION  (pure — no Plotly)
# ---------------------------------------------------------------------------

def _dedup_consecutive_pivots(indices: list, arr) -> list:
    """
    Remove duplicate pivots caused by flat candles (identical adjacent values).

    When two consecutive candles share the same high/low, both pass the == min/max
    check and both get added as pivots. This keeps only the first of each such run,
    so divergence lines always anchor to a single well-defined pivot.
    """
    if not indices:
        return indices
    result = [indices[0]]
    for i in range(1, len(indices)):
        prev_idx = indices[i - 1]
        curr_idx = indices[i]
        # Skip if this index is a direct continuation of the previous flat run
        if curr_idx == prev_idx + 1 and arr[curr_idx] == arr[prev_idx]:
            continue
        result.append(curr_idx)
    return result


def find_pivot_indices(
    series: pd.Series,
    window: int = PIVOT_WINDOW,
    left_bars: Optional[int] = None,
    right_bars: Optional[int] = None,
) -> Tuple[list, list]:
    """
    Find pivot lows and highs in a Series.

    A pivot LOW  at index i: series[i] is the minimum within [i-left, i+right]
    A pivot HIGH at index i: series[i] is the maximum within [i-left, i+right]

    left_bars / right_bars override the symmetric `window` when provided.
    Returns (low_indices, high_indices).
    """
    lb = left_bars  if left_bars  is not None else window
    rb = right_bars if right_bars is not None else window
    lows, highs = [], []
    arr = series.values
    for i in range(lb, len(arr) - rb):
        window_slice = arr[i - lb: i + rb + 1]
        if arr[i] == window_slice.min():
            lows.append(i)
        if arr[i] == window_slice.max():
            highs.append(i)
    lows  = _dedup_consecutive_pivots(lows,  arr)
    highs = _dedup_consecutive_pivots(highs, arr)
    return lows, highs


def compute_divergence_score(
    price_from: float,
    price_to:   float,
    cvd_from:   float,
    cvd_to:     float,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute display-only percentage moves for price and CVD between two pivots.

    Returns (price_move_pct, cvd_move_pct):
      - price_move_pct: % change in price between the two pivots
      - cvd_move_pct:   % change in CVD between the two pivots (None if cvd_from ≈ 0)

    Used for alert message display only — not for signal filtering.
    """
    price_move_pct = (price_to - price_from) / price_from * 100 if price_from else None

    if abs(cvd_from) < 1e-6:
        return price_move_pct, None

    cvd_move_pct = (cvd_to - cvd_from) / abs(cvd_from) * 100
    return price_move_pct, cvd_move_pct


def compute_quality_score(
    merged:     pd.DataFrame,
    from_idx:   int,
    to_idx:     int,
    price_col:  str,
    std_window: int = 200,
) -> Tuple[float, float, float, int]:
    """
    Compute raw signal quality dimensions for backtesting and analysis.

    1. Persistence: fraction of bars in the pivot window [from_idx..to_idx]
       where price (close) and CVD (close) moved in opposite directions.
       This is the primary filter criterion (threshold: > 0.5).

    2. price_atr_ratio: pivot-to-pivot price move divided by ATR(14).
       Raw value — e.g. 2.1 means the move was 2.1× a typical candle range.

    3. cvd_sigma: pivot-to-pivot CVD move divided by the expected random walk
       range (rolling_std × sqrt(window_bars)). Raw value — e.g. 3.4 means
       the CVD move was 3.4× what random noise would produce over this window.

    Returns (persistence, price_atr_ratio, cvd_sigma, window_bars).
    All values are raw — no normalization or composite weighting applied.
    Weights and thresholds are determined after empirical backtesting.
    """
    import math

    window_len = to_idx - from_idx  # number of bars in pivot window

    # --- Persistence ---
    if window_len < 2:
        persistence = 0.0
    else:
        w           = merged.iloc[from_idx: to_idx + 1]
        price_delta = w["p_close"].diff().iloc[1:]
        cvd_delta   = w["cvd_close"].diff().iloc[1:]

        valid   = (price_delta != 0) & (cvd_delta != 0)
        n_valid = int(valid.sum())
        if n_valid == 0:
            persistence = 0.0
        else:
            diverge     = ((price_delta > 0) != (cvd_delta > 0)) & valid
            persistence = float(diverge.sum()) / n_valid

    # --- Price move / ATR(14) ---
    atr_slice      = merged.iloc[max(0, to_idx - 13): to_idx + 1]
    atr            = float((atr_slice["p_high"] - atr_slice["p_low"]).mean())
    price_move     = abs(float(merged[price_col].iloc[to_idx]) - float(merged[price_col].iloc[from_idx]))
    price_atr_ratio = price_move / atr if atr > 1e-9 else 0.0

    # --- CVD move / expected random walk range ---
    cvd_hist  = merged["cvd_close"].iloc[max(0, to_idx - std_window): to_idx + 1]
    cvd_std   = float(cvd_hist.diff().dropna().std())
    cvd_move  = abs(float(merged["cvd_close"].iloc[to_idx]) - float(merged["cvd_close"].iloc[from_idx]))
    expected_cvd = cvd_std * math.sqrt(max(window_len, 1))
    cvd_sigma = cvd_move / expected_cvd if expected_cvd > 1e-9 else 0.0

    return persistence, price_atr_ratio, cvd_sigma, window_len


def detect_spot_signals(
    price_df:     pd.DataFrame,
    cvd_spot_df:  pd.DataFrame,
    pivot_window: int = PIVOT_WINDOW,
    cvd_mode:     str = "candle",
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Detect active divergence signals between price and CVD spot.

    cvd_mode: "candle" uses cvd_high/cvd_low at price pivot points;
              "line"   uses cvd_close at price pivot points.

    Returns (low_data, high_data). Each is None or a dict with keys:
      signal, price_from, price_to, cvd_from, cvd_to, timestamp
    """
    if price_df.empty or cvd_spot_df.empty:
        return None, None

    merged = pd.merge(
        price_df[["timestamp", "open", "high", "low", "close"]].rename(
            columns={"high": "p_high", "low": "p_low", "close": "p_close", "open": "p_open"}
        ),
        cvd_spot_df[["timestamp", "cvd_high", "cvd_low", "cvd_close"]],
        on="timestamp", how="inner",
    ).reset_index(drop=True)

    if len(merged) < pivot_window * 2 + 2:
        return None, None

    # CVD column to read at price pivot points
    cvd_lo_col = "cvd_close" if cvd_mode == "line" else "cvd_low"
    cvd_hi_col = "cvd_close" if cvd_mode == "line" else "cvd_high"

    p_lows, _  = find_pivot_indices(merged["p_low"],  pivot_window)
    _, p_highs = find_pivot_indices(merged["p_high"], pivot_window)
    curr = len(merged) - 1

    low_data:  Optional[dict] = None
    high_data: Optional[dict] = None

    # Check low: current candle is the lowest since last confirmed pivot low
    if p_lows:
        last_l = p_lows[-1]
        since = merged["p_low"].iloc[last_l + 1: curr + 1]
        if not since.empty and since.min() == merged["p_low"].iloc[curr]:
            l_pdif = merged["p_low"].iloc[curr]         - merged["p_low"].iloc[last_l]
            l_cdif = merged[cvd_lo_col].iloc[curr]      - merged[cvd_lo_col].iloc[last_l]
            signal = ""
            if   l_pdif < 0 and l_cdif > 0: signal = "SELLING EXHAUSTION"
            elif l_pdif > 0 and l_cdif < 0: signal = "SELLING ABSORPTION"
            if signal:
                pf = float(merged["p_low"].iloc[last_l])
                pt = float(merged["p_low"].iloc[curr])
                cf = float(merged[cvd_lo_col].iloc[last_l])
                ct = float(merged[cvd_lo_col].iloc[curr])
                p_pct, c_pct = compute_divergence_score(pf, pt, cf, ct)
                persistence, price_atr_ratio, cvd_sigma, window_bars = compute_quality_score(
                    merged, last_l, curr, "p_low"
                )
                low_data = {
                    "signal":          signal,
                    "price_from":      pf,
                    "price_to":        pt,
                    "cvd_from":        cf,
                    "cvd_to":          ct,
                    "timestamp":       merged["timestamp"].iloc[curr],
                    "pivot_from_ts":   merged["timestamp"].iloc[last_l],
                    "price_move_pct":  p_pct,
                    "cvd_move_pct":    c_pct,
                    "persistence":     persistence,
                    "price_atr_ratio": price_atr_ratio,
                    "cvd_sigma":       cvd_sigma,
                    "window_bars":     window_bars,
                    "div_score":       persistence,  # filter criterion
                }

    # Check high: current candle is the highest since last confirmed pivot high
    if p_highs:
        last_h = p_highs[-1]
        since = merged["p_high"].iloc[last_h + 1: curr + 1]
        if not since.empty and since.max() == merged["p_high"].iloc[curr]:
            h_pdif = merged["p_high"].iloc[curr]        - merged["p_high"].iloc[last_h]
            h_cdif = merged[cvd_hi_col].iloc[curr]      - merged[cvd_hi_col].iloc[last_h]
            signal = ""
            if   h_pdif > 0 and h_cdif < 0: signal = "BUYING EXHAUSTION"
            elif h_pdif < 0 and h_cdif > 0: signal = "BUYING ABSORPTION"
            if signal:
                pf = float(merged["p_high"].iloc[last_h])
                pt = float(merged["p_high"].iloc[curr])
                cf = float(merged[cvd_hi_col].iloc[last_h])
                ct = float(merged[cvd_hi_col].iloc[curr])
                p_pct, c_pct = compute_divergence_score(pf, pt, cf, ct)
                persistence, price_atr_ratio, cvd_sigma, window_bars = compute_quality_score(
                    merged, last_h, curr, "p_high"
                )
                high_data = {
                    "signal":          signal,
                    "price_from":      pf,
                    "price_to":        pt,
                    "cvd_from":        cf,
                    "cvd_to":          ct,
                    "timestamp":       merged["timestamp"].iloc[curr],
                    "pivot_from_ts":   merged["timestamp"].iloc[last_h],
                    "price_move_pct":  p_pct,
                    "cvd_move_pct":    c_pct,
                    "persistence":     persistence,
                    "price_atr_ratio": price_atr_ratio,
                    "cvd_sigma":       cvd_sigma,
                    "window_bars":     window_bars,
                    "div_score":       persistence,  # filter criterion
                }

    return low_data, high_data


# ---------------------------------------------------------------------------
# FIGURE BUILDING  (Plotly — no Dash)
# ---------------------------------------------------------------------------

def build_figure(
    price_df:         pd.DataFrame,
    cvd_spot_df:      pd.DataFrame,
    cvd_futures_df:   pd.DataFrame,
    oi_df:            pd.DataFrame,
    show_divergences: bool = True,
    interval_str:     str  = "15m",
    show_pivots:      bool = False,
    pivot_left:       int  = PIVOT_WINDOW,
    pivot_right:      int  = PIVOT_WINDOW,
    cvd_spot_mode:    str  = "candle",
    cvd_futures_mode: str  = "candle",
    oi_mode:          str  = "candle",
) -> Tuple[go.Figure, List[dict]]:
    """
    Assemble the 5-panel Plotly figure.
    Returns (fig, active_signal_data) where active_signal_data is a list
    of signal dicts from detect_spot_signals (used for Telegram alerts).

    cvd_spot_mode / cvd_futures_mode / oi_mode: "candle" or "line"
    """

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.36, 0.18, 0.18, 0.18, 0.10],
    )

    def candlestick(df, x, o, h, l, c, name, row):
        fig.add_trace(go.Candlestick(
            x=df[x], open=df[o], high=df[h], low=df[l], close=df[c],
            name=name,
            increasing_line_color=CANDLE_UP,   increasing_fillcolor=CANDLE_UP,
            decreasing_line_color=CANDLE_DOWN, decreasing_fillcolor=CANDLE_DOWN,
            line_width=1,
        ), row=row, col=1)

    def linechart(df, x, y, name, row, color="#26a69a"):
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y],
            mode="lines",
            name=name,
            line=dict(color=color, width=1.5),
        ), row=row, col=1)

    def spike_trigger(df, x, row):
        """Invisible scatter trace that triggers crosshair propagation across panels.
        go.Candlestick and go.Bar don't propagate spike lines to linked axes — Scatter does."""
        fig.add_trace(go.Scatter(
            x=df[x], y=[None] * len(df),
            mode="markers",
            marker=dict(opacity=0, size=1),
            hoverinfo="skip",
            showlegend=False,
        ), row=row, col=1)

    # Panel 1 — Price
    if not price_df.empty:
        candlestick(price_df, "timestamp", "open", "high", "low", "close", "BTC/USDT", 1)
        spike_trigger(price_df, "timestamp", 1)

        # Current price: horizontal dashed line + label on the right y-axis
        last_price = price_df["close"].iloc[-1]
        fig.add_hline(
            y=last_price,
            line_dash="dot",
            line_color="rgba(255,255,255,0.35)",
            line_width=1,
            row=1, col=1,
        )
        fig.add_annotation(
            x=1, xref="paper",
            y=last_price, yref="y",
            text=f"<b>{last_price:,.2f}</b>",
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(40,40,40,0.92)",
            bordercolor="rgba(255,255,255,0.25)",
            borderpad=4,
            xanchor="left",
        )

    # Panel 2 — CVD Spot
    if not cvd_spot_df.empty:
        if cvd_spot_mode == "line":
            linechart(cvd_spot_df, "timestamp", "cvd_close", "CVD Spot", 2, color="white")
        else:
            candlestick(cvd_spot_df, "timestamp", "cvd_open", "cvd_high", "cvd_low", "cvd_close", "CVD Spot", 2)
            spike_trigger(cvd_spot_df, "timestamp", 2)

    # Panel 3 — CVD Futures
    if not cvd_futures_df.empty:
        if cvd_futures_mode == "line":
            linechart(cvd_futures_df, "timestamp", "cvd_close", "CVD Futures", 3, color="#ab47bc")
        else:
            candlestick(cvd_futures_df, "timestamp", "cvd_open", "cvd_high", "cvd_low", "cvd_close", "CVD Futures", 3)
            spike_trigger(cvd_futures_df, "timestamp", 3)

    # Panel 4 — OI
    if not oi_df.empty:
        if oi_mode == "line":
            linechart(oi_df, "timestamp", "close", "Open Interest", 4, color="#ef5350")
        else:
            candlestick(oi_df, "timestamp", "open", "high", "low", "close", "Open Interest", 4)
            spike_trigger(oi_df, "timestamp", 4)

    # Panel 5 — Delta Spot
    if not cvd_spot_df.empty:
        delta = cvd_spot_df["cvd_close"] - cvd_spot_df["cvd_open"]
        bar_colors = [CANDLE_UP if d >= 0 else CANDLE_DOWN for d in delta]
        fig.add_trace(go.Bar(
            x=cvd_spot_df["timestamp"],
            y=delta,
            marker_color=bar_colors,
            marker_line_width=0,
            name="Delta Spot",
        ), row=5, col=1)
        spike_trigger(cvd_spot_df, "timestamp", 5)

    # ── Divergences ───────────────────────────────────────────────────────
    active_signals     = []   # (color, label) for banner
    active_signal_data = []   # full dicts for alert system

    if show_divergences and not price_df.empty and not cvd_spot_df.empty:
        cvd_lo_col = "cvd_close" if cvd_spot_mode == "line" else "cvd_low"
        cvd_hi_col = "cvd_close" if cvd_spot_mode == "line" else "cvd_high"

        merged = pd.merge(
            price_df[["timestamp", "high", "low"]].rename(
                columns={"high": "p_high", "low": "p_low"}
            ),
            cvd_spot_df[["timestamp", "cvd_high", "cvd_low", "cvd_close"]],
            on="timestamp", how="inner",
        ).reset_index(drop=True)

        div_lb = pivot_left
        div_rb = pivot_right
        if len(merged) >= div_lb + div_rb + 2:
            p_lows, _  = find_pivot_indices(merged["p_low"],  left_bars=div_lb, right_bars=div_rb)
            _, p_highs = find_pivot_indices(merged["p_high"], left_bars=div_lb, right_bars=div_rb)
            curr = len(merged) - 1

            # Offset for CVD divergence lines: 15% of visible CVD range,
            # applied only to the LEFT (older) endpoint. The RIGHT (newer) endpoint
            # touches the CVD curve exactly. Lines "peel away" from CVD at the far end.
            cvd_vals   = merged["cvd_close"]
            cvd_offset = (cvd_vals.max() - cvd_vals.min()) * 0.15

            def draw_line(x0, y0, x1, y1, color, dash, width, row):
                fig.add_shape(
                    type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color=color, width=width, dash=dash),
                    row=row, col=1,
                )

            # Historical low divergences (always drawn)
            for i in range(1, len(p_lows)):
                p1, p2 = p_lows[i - 1], p_lows[i]
                pdif = merged["p_low"].iloc[p2]        - merged["p_low"].iloc[p1]
                cdif = merged[cvd_lo_col].iloc[p2]     - merged[cvd_lo_col].iloc[p1]
                if   pdif < 0 and cdif > 0: color, dash = "#2196f3", "dash"
                elif pdif > 0 and cdif < 0: color, dash = "#2196f3", "solid"
                else: continue
                t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
                draw_line(t0, merged["p_low"].iloc[p1],                       t1, merged["p_low"].iloc[p2],                  color, dash, 1.5, 1)
                draw_line(t0, merged[cvd_lo_col].iloc[p1] - cvd_offset,       t1, merged[cvd_lo_col].iloc[p2],               color, dash, 1.5, 2)

            # Historical high divergences (always drawn)
            for i in range(1, len(p_highs)):
                p1, p2 = p_highs[i - 1], p_highs[i]
                pdif = merged["p_high"].iloc[p2]       - merged["p_high"].iloc[p1]
                cdif = merged[cvd_hi_col].iloc[p2]     - merged[cvd_hi_col].iloc[p1]
                if   pdif > 0 and cdif < 0: color, dash = "orange", "dash"
                elif pdif < 0 and cdif > 0: color, dash = "orange", "solid"
                else: continue
                t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
                draw_line(t0, merged["p_high"].iloc[p1],                      t1, merged["p_high"].iloc[p2],                 color, dash, 1.5, 1)
                draw_line(t0, merged[cvd_hi_col].iloc[p1] + cvd_offset,       t1, merged[cvd_hi_col].iloc[p2],               color, dash, 1.5, 2)

            # Live signal — current candle vs last confirmed pivot
            low_data, high_data = detect_spot_signals(price_df, cvd_spot_df, cvd_mode=cvd_spot_mode)
            for data in (low_data, high_data):
                if not data:
                    continue
                color = "#2196f3" if "SELLER" in data["signal"] else "orange"
                dash  = "dash" if "EXHAUSTION" in data["signal"] else "solid"
                is_low = "SELLER" in data["signal"]
                p_col  = "p_low"    if is_low else "p_high"
                c_col  = cvd_lo_col if is_low else cvd_hi_col
                sign   = -1         if is_low else +1   # below for blue, above for orange
                pivots = p_lows if is_low else p_highs
                if pivots:
                    last_p = pivots[-1]
                    t0, t1 = merged["timestamp"].iloc[last_p], merged["timestamp"].iloc[curr]
                    draw_line(t0, merged[p_col].iloc[last_p], t1, merged[p_col].iloc[curr], color, dash, 3, 1)
                    draw_line(t0, merged[c_col].iloc[last_p] + sign * cvd_offset, t1, merged[c_col].iloc[curr], color, dash, 3, 2)
                p_pct = data.get("price_move_pct")
                c_pct = data.get("cvd_move_pct")
                score = data.get("div_score")
                metrics = ""
                if p_pct is not None and c_pct is not None:
                    score_str = f"  Δ{score:.1f}%" if score is not None else ""
                    metrics = f"  ·  price {p_pct:+.2f}%  CVD {c_pct:+.2f}%{score_str}"
                active_signals.append((color, f"{data['signal']}{metrics}"))
                active_signal_data.append(data)

        # CVD Futures divergences disabled — futures data still accumulating

    # ── Pivot markers ─────────────────────────────────────────────────────
    if show_pivots and not price_df.empty:
        p_lows_vis, p_highs_vis = find_pivot_indices(
            price_df["low"],
            left_bars=pivot_left,
            right_bars=pivot_right,
        )
        _, p_highs_vis2 = find_pivot_indices(
            price_df["high"],
            left_bars=pivot_left,
            right_bars=pivot_right,
        )
        price_range  = price_df["high"].max() - price_df["low"].min()
        arrow_offset = price_range * 0.045  # ~4.5% of visible range
        if p_lows_vis:
            fig.add_trace(go.Scatter(
                x=price_df["timestamp"].iloc[p_lows_vis],
                y=price_df["low"].iloc[p_lows_vis] - arrow_offset,
                mode="markers",
                marker=dict(symbol="triangle-up", size=8, color="lime"),
                name="Pivot Low",
                hoverinfo="skip",
            ), row=1, col=1)
        if p_highs_vis2:
            fig.add_trace(go.Scatter(
                x=price_df["timestamp"].iloc[p_highs_vis2],
                y=price_df["high"].iloc[p_highs_vis2] + arrow_offset,
                mode="markers",
                marker=dict(symbol="triangle-down", size=8, color="#ef5350"),
                name="Pivot High",
                hoverinfo="skip",
            ), row=1, col=1)

    # Live signal banner — inside chart, top of panel 1
    if active_signals:
        signal_text    = "   |   ".join(f"<b>{label}</b>" for _, label in active_signals)
        dominant_color = active_signals[0][0]
        fig.add_annotation(
            x=0.5, y=1.0, xref="paper", yref="paper",
            text=signal_text, showarrow=False,
            font=dict(size=14, color=dominant_color),
            bgcolor="rgba(0,0,0,0.80)",
            bordercolor=dominant_color,
            borderpad=8,
            xanchor="center", yanchor="top",
        )

    # Panel labels — pinned inside each panel just below its top border
    panel_labels = [
        (1.000, "BTC/USDT  ·  Binance Spot"),
        (0.644, "CVD Spot  ·  Aggregated  · BTC"),
        (0.462, "CVD Futures  ·  Aggregated  · BTC"),
        (0.279, "Open Interest  ·  Binance Futures"),
        (0.096, "Delta Spot  ·  Aggregated  · BTC"),
    ]
    for y_paper, text in panel_labels:
        fig.add_annotation(
            x=0.005, y=y_paper, xref="paper", yref="paper",
            text=f"<b>{text}</b>", showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,0.65)"),
            align="left", xanchor="left", yanchor="top",
        )

    # X-axis range: extend right by 10 empty candles
    dtick = _NICE_TICK_MS[0]
    if not price_df.empty:
        candle_td = pd.Timedelta(seconds=INTERVAL_SECONDS.get(interval_str, 900))
        x_min     = price_df["timestamp"].iloc[0]
        x_max     = price_df["timestamp"].iloc[-1] + candle_td * 10
        fig.update_xaxes(range=[x_min, x_max])
        total_ms  = int((x_max - x_min).total_seconds() * 1000)
        dtick     = min(_NICE_TICK_MS, key=lambda v: abs(v - total_ms / 20))

    fig.update_xaxes(
        rangeslider_visible=False,
        gridcolor="#1e1e1e",
        tickformat="%H:%M\n%d/%m",
        dtick=dtick,
        tickfont=dict(color="rgba(255,255,255,0.45)"),
        showspikes=True,
        spikecolor="rgba(255,255,255,0.3)",
        spikethickness=1,
        spikemode="across",
        spikesnap="cursor",
    )
    fig.update_yaxes(
        side="right",
        gridcolor="#1e1e1e",
        tickfont=dict(color="rgba(255,255,255,0.45)"),
        showspikes=True,
        spikecolor="#555",
        spikethickness=1,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111",
        plot_bgcolor="#0d0d0d",
        autosize=True,
        margin=dict(l=0, r=70, t=5, b=0),
        showlegend=False,
        hovermode="x",
        hoversubplots="axis",
    )

    return fig, active_signal_data


# ---------------------------------------------------------------------------
# TIMEZONE HELPER
# ---------------------------------------------------------------------------

def to_warsaw(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """
    Return a copy of df with timestamp column converted to Europe/Warsaw.
    Timezone info is stripped after conversion so Plotly.js displays the
    local time as-is without converting back to UTC in the browser.
    """
    df = df.copy()
    df[ts_col] = df[ts_col].dt.tz_convert(_WARSAW).dt.tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# ALERT FIGURE  (mobile-optimised, 540×960)
# ---------------------------------------------------------------------------

def build_alert_figure(
    price_df:         pd.DataFrame,
    cvd_spot_df:      pd.DataFrame,
    cvd_futures_df:   pd.DataFrame,
    oi_df:            pd.DataFrame,
    interval_str:     str = "15m",
) -> go.Figure:
    """
    Build a mobile-optimised 4-panel figure for Telegram alert screenshots.

    Differences from build_figure():
    - Timestamps converted to Europe/Warsaw
    - Banner placed above the chart area (not overlapping candles)
    - Banner text without the 'SPOT ·' prefix
    - X-axis ticks: 8 per width (dynamic dtick)
    - Larger top margin to accommodate the banner
    """
    # Detect signals on UTC data BEFORE timezone conversion
    # (detect_spot_signals merges on timestamp — must stay UTC for consistency)
    low_data, high_data = detect_spot_signals(price_df, cvd_spot_df)

    # Convert timestamps to Warsaw time for display
    def _to_waw(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return to_warsaw(df)

    price_df       = _to_waw(price_df)
    cvd_spot_df    = _to_waw(cvd_spot_df)
    cvd_futures_df = _to_waw(cvd_futures_df)
    oi_df          = _to_waw(oi_df)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.01,
        row_heights=[0.40, 0.20, 0.20, 0.20],
    )

    def candlestick(df, x, o, h, l, c, name, row):
        fig.add_trace(go.Candlestick(
            x=df[x], open=df[o], high=df[h], low=df[l], close=df[c],
            name=name,
            increasing_line_color=CANDLE_UP,   increasing_fillcolor=CANDLE_UP,
            decreasing_line_color=CANDLE_DOWN, decreasing_fillcolor=CANDLE_DOWN,
            line_width=1,
        ), row=row, col=1)

    if not price_df.empty:
        candlestick(price_df, "timestamp", "open", "high", "low", "close", "BTC/USDT", 1)
        last_price = price_df["close"].iloc[-1]
        fig.add_hline(
            y=last_price, line_dash="dot",
            line_color="rgba(255,255,255,0.35)", line_width=1,
            row=1, col=1,
        )
        fig.add_annotation(
            x=1, xref="paper", y=last_price, yref="y",
            text=f"<b>{last_price:,.2f}</b>", showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(40,40,40,0.92)",
            bordercolor="rgba(255,255,255,0.25)",
            borderpad=4, xanchor="left",
        )

    if not cvd_spot_df.empty:
        candlestick(cvd_spot_df, "timestamp", "cvd_open", "cvd_high", "cvd_low", "cvd_close", "CVD Spot", 2)
    if not cvd_futures_df.empty:
        candlestick(cvd_futures_df, "timestamp", "cvd_open", "cvd_high", "cvd_low", "cvd_close", "CVD Futures", 3)
    if not oi_df.empty:
        candlestick(oi_df, "timestamp", "open", "high", "low", "close", "Open Interest", 4)

    # ── Divergences ───────────────────────────────────────────────────────
    active_signals = []

    if not price_df.empty and not cvd_spot_df.empty:
        merged = pd.merge(
            price_df[["timestamp", "high", "low"]].rename(
                columns={"high": "p_high", "low": "p_low"}
            ),
            cvd_spot_df[["timestamp", "cvd_high", "cvd_low"]],
            on="timestamp", how="inner",
        ).reset_index(drop=True)

        if len(merged) >= PIVOT_WINDOW * 2 + 2:
            p_lows, _  = find_pivot_indices(merged["p_low"])
            _, p_highs = find_pivot_indices(merged["p_high"])
            curr = len(merged) - 1

            cvd_offset = (cvd_spot_df["cvd_close"].max() - cvd_spot_df["cvd_close"].min()) * 0.25

            def draw_line(x0, y0, x1, y1, color, dash, width, row):
                fig.add_shape(
                    type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color=color, width=width, dash=dash),
                    row=row, col=1,
                )

            for i in range(1, len(p_lows)):
                p1, p2 = p_lows[i - 1], p_lows[i]
                pdif = merged["p_low"].iloc[p2]   - merged["p_low"].iloc[p1]
                cdif = merged["cvd_low"].iloc[p2] - merged["cvd_low"].iloc[p1]
                if   pdif < 0 and cdif > 0: color, dash = "#2196f3", "dash"
                elif pdif > 0 and cdif < 0: color, dash = "#2196f3", "solid"
                else: continue
                t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
                draw_line(t0, merged["p_low"].iloc[p1],                    t1, merged["p_low"].iloc[p2],               color, dash, 1.5, 1)
                draw_line(t0, merged["cvd_low"].iloc[p1] - cvd_offset,     t1, merged["cvd_low"].iloc[p2],             color, dash, 1.5, 2)

            for i in range(1, len(p_highs)):
                p1, p2 = p_highs[i - 1], p_highs[i]
                pdif = merged["p_high"].iloc[p2]   - merged["p_high"].iloc[p1]
                cdif = merged["cvd_high"].iloc[p2] - merged["cvd_high"].iloc[p1]
                if   pdif > 0 and cdif < 0: color, dash = "orange", "dash"
                elif pdif < 0 and cdif > 0: color, dash = "orange", "solid"
                else: continue
                t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
                draw_line(t0, merged["p_high"].iloc[p1],                   t1, merged["p_high"].iloc[p2],              color, dash, 1.5, 1)
                draw_line(t0, merged["cvd_high"].iloc[p1] + cvd_offset,    t1, merged["cvd_high"].iloc[p2],             color, dash, 1.5, 2)

            # Live signal — already detected on UTC data before timezone conversion
            for data in (low_data, high_data):
                if not data:
                    continue
                color = "#2196f3" if "SELLER" in data["signal"] else "orange"
                dash  = "dash" if "EXHAUSTION" in data["signal"] else "solid"
                is_low = "SELLER" in data["signal"]
                p_col, c_col = ("p_low", "cvd_low") if is_low else ("p_high", "cvd_high")
                sign   = -1 if is_low else +1
                pivots = p_lows if is_low else p_highs
                if pivots:
                    last_p = pivots[-1]
                    t0 = merged["timestamp"].iloc[last_p]
                    t1 = merged["timestamp"].iloc[curr]
                    draw_line(t0, merged[p_col].iloc[last_p], t1, merged[p_col].iloc[curr], color, dash, 3, 1)
                    draw_line(t0, merged[c_col].iloc[last_p] + sign * cvd_offset, t1, merged[c_col].iloc[curr], color, dash, 3, 2)

                p_pct = data.get("price_move_pct")
                c_pct = data.get("cvd_move_pct")
                score = data.get("div_score")
                metrics = ""
                if p_pct is not None and c_pct is not None:
                    score_str = f"  Δ{score:.1f}%" if score is not None else ""
                    metrics = f"  ·  price {p_pct:+.2f}%  CVD {c_pct:+.2f}%{score_str}"
                active_signals.append((color, f"{data['signal']}{metrics}"))

    # Banner — inside chart, top of panel 1
    if active_signals:
        signal_text    = "   |   ".join(f"<b>{label}</b>" for _, label in active_signals)
        dominant_color = active_signals[0][0]
        fig.add_annotation(
            x=0.5, y=1.0, xref="paper", yref="paper",
            text=signal_text, showarrow=False,
            font=dict(size=13, color=dominant_color),
            bgcolor="rgba(0,0,0,0.90)",
            bordercolor=dominant_color,
            borderpad=7,
            xanchor="center", yanchor="top",
        )

    # Panel labels — pinned inside each panel just below its top border
    panel_labels = [
        (1.0,   "BTC/USDT  ·  Binance Spot"),
        (0.602, "CVD Spot  ·  Aggregated"),
        (0.398, "CVD Futures  ·  Aggregated"),
        (0.194, "Open Interest  ·  Binance Futures"),
    ]
    for y_paper, text in panel_labels:
        fig.add_annotation(
            x=0.005, y=y_paper, xref="paper", yref="paper",
            text=f"<b>{text}</b>", showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,0.45)"),
            align="left", xanchor="left", yanchor="top",
        )

    # X-axis: 8 ticks across the width, Warsaw time format
    dtick = _NICE_TICK_MS[0]
    if not price_df.empty:
        candle_td = pd.Timedelta(seconds=INTERVAL_SECONDS.get(interval_str, 900))
        x_min     = price_df["timestamp"].iloc[0]
        x_max     = price_df["timestamp"].iloc[-1] + candle_td * 10
        fig.update_xaxes(range=[x_min, x_max])
        total_ms  = int((x_max - x_min).total_seconds() * 1000)
        dtick     = min(_NICE_TICK_MS, key=lambda v: abs(v - total_ms / 8))

    fig.update_xaxes(
        rangeslider_visible=False,
        gridcolor="#1e1e1e",
        tickformat="%H:%M\n%d/%m",
        dtick=dtick,
        tickfont=dict(color="rgba(255,255,255,0.45)", size=9),
        showspikes=False,
    )
    fig.update_yaxes(
        side="right",
        gridcolor="#1e1e1e",
        tickfont=dict(color="rgba(255,255,255,0.45)", size=9),
        showspikes=False,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111",
        plot_bgcolor="#0d0d0d",
        margin=dict(l=0, r=55, t=5, b=25),
        showlegend=False,
        hovermode=False,
    )

    return fig
