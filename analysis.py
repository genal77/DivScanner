"""
analysis.py — Shared data processing, divergence detection, and figure building.

Imported by both app.py (display) and collector.py (alerts).
No Dash dependencies — only pandas and plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Tuple

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
    "SELLERS EXHAUSTION": "🟢",
    "SELLERS ABSORPTION": "🔵",
    "BUYERS EXHAUSTION":  "🔴",
    "BUYERS ABSORPTION":  "🟠",
}

CANDLE_UP   = "#26a69a"
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


def compute_oi_ohlc(oi_df: pd.DataFrame, pandas_interval: str) -> pd.DataFrame:
    """Resample 1m OI snapshots into OHLC candles for the target interval."""
    if oi_df.empty:
        return pd.DataFrame()

    oi = oi_df.set_index("timestamp").sort_index()
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

def find_pivot_indices(series: pd.Series, window: int = PIVOT_WINDOW) -> Tuple[list, list]:
    """
    Find pivot lows and highs in a Series.

    A pivot LOW  at index i: series[i] is the minimum within [i-window, i+window]
    A pivot HIGH at index i: series[i] is the maximum within [i-window, i+window]

    Returns (low_indices, high_indices).
    First and last `window` candles cannot be pivots by definition.
    """
    lows, highs = [], []
    arr = series.values
    for i in range(window, len(arr) - window):
        window_slice = arr[i - window: i + window + 1]
        if arr[i] == window_slice.min():
            lows.append(i)
        if arr[i] == window_slice.max():
            highs.append(i)
    return lows, highs


def detect_spot_signals(
    price_df:     pd.DataFrame,
    cvd_spot_df:  pd.DataFrame,
    pivot_window: int = PIVOT_WINDOW,
) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Detect active divergence signals between price and CVD spot.

    Returns (low_data, high_data). Each is None or a dict with keys:
      signal, price_from, price_to, cvd_from, cvd_to, timestamp
    """
    if price_df.empty or cvd_spot_df.empty:
        return None, None

    merged = pd.merge(
        price_df[["timestamp", "high", "low"]].rename(
            columns={"high": "p_high", "low": "p_low"}
        ),
        cvd_spot_df[["timestamp", "cvd_high", "cvd_low"]],
        on="timestamp", how="inner",
    ).reset_index(drop=True)

    if len(merged) < pivot_window * 2 + 2:
        return None, None

    p_lows, p_highs = find_pivot_indices(merged["p_low"], pivot_window)
    curr = len(merged) - 1

    low_data:  Optional[dict] = None
    high_data: Optional[dict] = None

    # Check low: current candle is the lowest since last confirmed pivot low
    if p_lows:
        last_l = p_lows[-1]
        since = merged["p_low"].iloc[last_l + 1: curr + 1]
        if not since.empty and since.min() == merged["p_low"].iloc[curr]:
            l_pdif = merged["p_low"].iloc[curr]   - merged["p_low"].iloc[last_l]
            l_cdif = merged["cvd_low"].iloc[curr] - merged["cvd_low"].iloc[last_l]
            signal = ""
            if   l_pdif < 0 and l_cdif > 0: signal = "SELLERS EXHAUSTION"
            elif l_pdif > 0 and l_cdif < 0: signal = "SELLERS ABSORPTION"
            if signal:
                low_data = {
                    "signal":     signal,
                    "price_from": float(merged["p_low"].iloc[last_l]),
                    "price_to":   float(merged["p_low"].iloc[curr]),
                    "cvd_from":   float(merged["cvd_low"].iloc[last_l]),
                    "cvd_to":     float(merged["cvd_low"].iloc[curr]),
                    "timestamp":  merged["timestamp"].iloc[curr],
                }

    # Check high: current candle is the highest since last confirmed pivot high
    if p_highs:
        last_h = p_highs[-1]
        since = merged["p_high"].iloc[last_h + 1: curr + 1]
        if not since.empty and since.max() == merged["p_high"].iloc[curr]:
            h_pdif = merged["p_high"].iloc[curr]    - merged["p_high"].iloc[last_h]
            h_cdif = merged["cvd_high"].iloc[curr]  - merged["cvd_high"].iloc[last_h]
            signal = ""
            if   h_pdif > 0 and h_cdif < 0: signal = "BUYERS EXHAUSTION"
            elif h_pdif < 0 and h_cdif > 0: signal = "BUYERS ABSORPTION"
            if signal:
                high_data = {
                    "signal":     signal,
                    "price_from": float(merged["p_high"].iloc[last_h]),
                    "price_to":   float(merged["p_high"].iloc[curr]),
                    "cvd_from":   float(merged["cvd_high"].iloc[last_h]),
                    "cvd_to":     float(merged["cvd_high"].iloc[curr]),
                    "timestamp":  merged["timestamp"].iloc[curr],
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
) -> Tuple[go.Figure, List[dict]]:
    """
    Assemble the 4-panel Plotly figure.
    Returns (fig, active_signal_data) where active_signal_data is a list
    of signal dicts from detect_spot_signals (used for Telegram alerts).
    """

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
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

    # Panel 1 — Price
    if not price_df.empty:
        candlestick(price_df, "timestamp", "open", "high", "low", "close", "BTC/USDT", 1)

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
        candlestick(cvd_spot_df, "timestamp", "cvd_open", "cvd_high", "cvd_low", "cvd_close", "CVD Spot", 2)

    # Panel 3 — CVD Futures
    if not cvd_futures_df.empty:
        candlestick(cvd_futures_df, "timestamp", "cvd_open", "cvd_high", "cvd_low", "cvd_close", "CVD Futures", 3)

    # Panel 4 — OI
    if not oi_df.empty:
        candlestick(oi_df, "timestamp", "open", "high", "low", "close", "Open Interest", 4)

    # ── Divergences ───────────────────────────────────────────────────────
    active_signals     = []   # (color, label) for banner
    active_signal_data = []   # full dicts for alert system

    if show_divergences:
        low_data, high_data = detect_spot_signals(price_df, cvd_spot_df)
        for data in (low_data, high_data):
            if data:
                color = "cyan" if "SELLER" in data["signal"] else "magenta"
                active_signals.append((color, f"SPOT · {data['signal']}"))
                active_signal_data.append(data)

                # Draw divergence lines on price panel (1) and CVD panel (2)
                sig = data["signal"]
                line_color = "cyan" if "SELLER" in sig else "magenta"
                line_dash  = "dash" if "EXHAUSTION" in sig else "solid"

                is_low = "SELLER" in sig
                p_col  = "p_low"  if is_low else "p_high"
                c_col  = "cvd_low" if is_low else "cvd_high"

                # Re-merge to get timestamps for drawing
                if not price_df.empty and not cvd_spot_df.empty:
                    merged = pd.merge(
                        price_df[["timestamp", "high", "low"]].rename(
                            columns={"high": "p_high", "low": "p_low"}
                        ),
                        cvd_spot_df[["timestamp", "cvd_high", "cvd_low"]],
                        on="timestamp", how="inner",
                    ).reset_index(drop=True)

                    p_lows, p_highs = find_pivot_indices(merged["p_low"])
                    pivots = p_lows if is_low else p_highs

                    # Historical lines
                    for i in range(1, len(pivots)):
                        p1, p2  = pivots[i - 1], pivots[i]
                        pdif    = merged[p_col].iloc[p2]  - merged[p_col].iloc[p1]
                        cdif    = merged[c_col].iloc[p2]  - merged[c_col].iloc[p1]
                        if is_low:
                            match = (pdif < 0 and cdif > 0) or (pdif > 0 and cdif < 0)
                            h_color = "cyan"
                            h_dash  = "dash" if (pdif < 0 and cdif > 0) else "solid"
                        else:
                            match = (pdif > 0 and cdif < 0) or (pdif < 0 and cdif > 0)
                            h_color = "magenta"
                            h_dash  = "dash" if (pdif > 0 and cdif < 0) else "solid"
                        if not match:
                            continue
                        t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
                        for row, col in [(1, p_col), (2, c_col)]:
                            fig.add_shape(
                                type="line",
                                x0=t0, y0=merged[col].iloc[p1],
                                x1=t1, y1=merged[col].iloc[p2],
                                line=dict(color=h_color, width=1.5, dash=h_dash),
                                row=row, col=1,
                            )

                    # Live signal line
                    curr   = len(merged) - 1
                    pivots_list = p_lows if is_low else p_highs
                    if pivots_list:
                        last_p = pivots_list[-1]
                        t0, t1 = merged["timestamp"].iloc[last_p], merged["timestamp"].iloc[curr]
                        for row, col in [(1, p_col), (2, c_col)]:
                            fig.add_shape(
                                type="line",
                                x0=t0, y0=merged[col].iloc[last_p],
                                x1=t1, y1=merged[col].iloc[curr],
                                line=dict(color=line_color, width=3, dash=line_dash),
                                row=row, col=1,
                            )

        # CVD Futures divergences disabled — futures data still accumulating

    # Live signal banner
    if active_signals:
        signal_text    = "   |   ".join(f"<b>{label}</b>" for _, label in active_signals)
        dominant_color = active_signals[0][0]
        fig.add_annotation(
            x=0.5, y=0.995, xref="paper", yref="paper",
            text=signal_text, showarrow=False,
            font=dict(size=14, color=dominant_color),
            bgcolor="rgba(0,0,0,0.80)",
            bordercolor=dominant_color,
            borderpad=8,
            xanchor="center", yanchor="top",
        )

    # Panel labels
    panel_labels = [
        (0.985, "BTC/USDT  ·  Binance Spot"),
        (0.570, "CVD Spot  ·  Aggregated"),
        (0.360, "CVD Futures  ·  Aggregated"),
        (0.145, "Open Interest  ·  Binance Futures"),
    ]
    for y_paper, text in panel_labels:
        fig.add_annotation(
            x=0.005, y=y_paper, xref="paper", yref="paper",
            text=f"<b>{text}</b>", showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,0.45)"),
            align="left", xanchor="left",
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
        spikecolor="#555",
        spikethickness=1,
        spikemode="across",
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
        margin=dict(l=0, r=70, t=0, b=0),
        showlegend=False,
        hovermode="x unified",
    )

    return fig, active_signal_data
