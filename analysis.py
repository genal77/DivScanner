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


def compute_divergence_score(
    price_from: float,
    price_to:   float,
    cvd_from:   float,
    cvd_to:     float,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute divergence strength metrics.

    Returns (price_move_pct, cvd_move_pct, div_score):
      - price_move_pct: % change in price between the two pivots
      - cvd_move_pct:   % change in CVD between the two pivots (None if cvd_from ≈ 0)
      - div_score:      abs gap between the two percentages (None if cvd_move_pct is None)

    div_score accumulates over time to build an empirical scale of signal strength.
    """
    price_move_pct = (price_to - price_from) / price_from * 100 if price_from else None

    if abs(cvd_from) < 1e-6:
        return price_move_pct, None, None

    cvd_move_pct = (cvd_to - cvd_from) / abs(cvd_from) * 100
    div_score = (
        abs(price_move_pct - cvd_move_pct)
        if price_move_pct is not None
        else None
    )
    return price_move_pct, cvd_move_pct, div_score


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
                pf = float(merged["p_low"].iloc[last_l])
                pt = float(merged["p_low"].iloc[curr])
                cf = float(merged["cvd_low"].iloc[last_l])
                ct = float(merged["cvd_low"].iloc[curr])
                p_pct, c_pct, score = compute_divergence_score(pf, pt, cf, ct)
                low_data = {
                    "signal":         signal,
                    "price_from":     pf,
                    "price_to":       pt,
                    "cvd_from":       cf,
                    "cvd_to":         ct,
                    "timestamp":      merged["timestamp"].iloc[curr],
                    "price_move_pct": p_pct,
                    "cvd_move_pct":   c_pct,
                    "div_score":      score,
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
                pf = float(merged["p_high"].iloc[last_h])
                pt = float(merged["p_high"].iloc[curr])
                cf = float(merged["cvd_high"].iloc[last_h])
                ct = float(merged["cvd_high"].iloc[curr])
                p_pct, c_pct, score = compute_divergence_score(pf, pt, cf, ct)
                high_data = {
                    "signal":         signal,
                    "price_from":     pf,
                    "price_to":       pt,
                    "cvd_from":       cf,
                    "cvd_to":         ct,
                    "timestamp":      merged["timestamp"].iloc[curr],
                    "price_move_pct": p_pct,
                    "cvd_move_pct":   c_pct,
                    "div_score":      score,
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

    if show_divergences and not price_df.empty and not cvd_spot_df.empty:
        merged = pd.merge(
            price_df[["timestamp", "high", "low"]].rename(
                columns={"high": "p_high", "low": "p_low"}
            ),
            cvd_spot_df[["timestamp", "cvd_high", "cvd_low"]],
            on="timestamp", how="inner",
        ).reset_index(drop=True)

        if len(merged) >= PIVOT_WINDOW * 2 + 2:
            p_lows, p_highs = find_pivot_indices(merged["p_low"])
            curr = len(merged) - 1

            def draw_line(x0, y0, x1, y1, color, dash, width, row):
                fig.add_shape(
                    type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                    line=dict(color=color, width=width, dash=dash),
                    row=row, col=1,
                )

            # Historical low divergences (always drawn)
            for i in range(1, len(p_lows)):
                p1, p2 = p_lows[i - 1], p_lows[i]
                pdif = merged["p_low"].iloc[p2]   - merged["p_low"].iloc[p1]
                cdif = merged["cvd_low"].iloc[p2] - merged["cvd_low"].iloc[p1]
                if   pdif < 0 and cdif > 0: color, dash = "cyan", "dash"
                elif pdif > 0 and cdif < 0: color, dash = "cyan", "solid"
                else: continue
                t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
                draw_line(t0, merged["p_low"].iloc[p1],   t1, merged["p_low"].iloc[p2],   color, dash, 1.5, 1)
                draw_line(t0, merged["cvd_low"].iloc[p1], t1, merged["cvd_low"].iloc[p2], color, dash, 1.5, 2)

            # Historical high divergences (always drawn)
            for i in range(1, len(p_highs)):
                p1, p2 = p_highs[i - 1], p_highs[i]
                pdif = merged["p_high"].iloc[p2]    - merged["p_high"].iloc[p1]
                cdif = merged["cvd_high"].iloc[p2]  - merged["cvd_high"].iloc[p1]
                if   pdif > 0 and cdif < 0: color, dash = "magenta", "dash"
                elif pdif < 0 and cdif > 0: color, dash = "magenta", "solid"
                else: continue
                t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
                draw_line(t0, merged["p_high"].iloc[p1],   t1, merged["p_high"].iloc[p2],   color, dash, 1.5, 1)
                draw_line(t0, merged["cvd_high"].iloc[p1], t1, merged["cvd_high"].iloc[p2], color, dash, 1.5, 2)

            # Live signal — current candle vs last confirmed pivot
            low_data, high_data = detect_spot_signals(price_df, cvd_spot_df)
            for data in (low_data, high_data):
                if not data:
                    continue
                color = "cyan" if "SELLER" in data["signal"] else "magenta"
                dash  = "dash" if "EXHAUSTION" in data["signal"] else "solid"
                is_low = "SELLER" in data["signal"]
                p_col, c_col = ("p_low", "cvd_low") if is_low else ("p_high", "cvd_high")
                pivots = p_lows if is_low else p_highs
                if pivots:
                    last_p = pivots[-1]
                    t0, t1 = merged["timestamp"].iloc[last_p], merged["timestamp"].iloc[curr]
                    draw_line(t0, merged[p_col].iloc[last_p], t1, merged[p_col].iloc[curr], color, dash, 3, 1)
                    draw_line(t0, merged[c_col].iloc[last_p], t1, merged[c_col].iloc[curr], color, dash, 3, 2)
                p_pct = data.get("price_move_pct")
                c_pct = data.get("cvd_move_pct")
                score = data.get("div_score")
                metrics = ""
                if p_pct is not None and c_pct is not None:
                    score_str = f"  Δ{score:.1f}%" if score is not None else ""
                    metrics = f"  ·  price {p_pct:+.2f}%  CVD {c_pct:+.2f}%{score_str}"
                active_signals.append((color, f"SPOT · {data['signal']}{metrics}"))
                active_signal_data.append(data)

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
            p_lows, p_highs = find_pivot_indices(merged["p_low"])
            curr = len(merged) - 1

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
                if   pdif < 0 and cdif > 0: color, dash = "cyan", "dash"
                elif pdif > 0 and cdif < 0: color, dash = "cyan", "solid"
                else: continue
                t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
                draw_line(t0, merged["p_low"].iloc[p1],   t1, merged["p_low"].iloc[p2],   color, dash, 1.5, 1)
                draw_line(t0, merged["cvd_low"].iloc[p1], t1, merged["cvd_low"].iloc[p2], color, dash, 1.5, 2)

            for i in range(1, len(p_highs)):
                p1, p2 = p_highs[i - 1], p_highs[i]
                pdif = merged["p_high"].iloc[p2]   - merged["p_high"].iloc[p1]
                cdif = merged["cvd_high"].iloc[p2] - merged["cvd_high"].iloc[p1]
                if   pdif > 0 and cdif < 0: color, dash = "magenta", "dash"
                elif pdif < 0 and cdif > 0: color, dash = "magenta", "solid"
                else: continue
                t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
                draw_line(t0, merged["p_high"].iloc[p1],   t1, merged["p_high"].iloc[p2],   color, dash, 1.5, 1)
                draw_line(t0, merged["cvd_high"].iloc[p1], t1, merged["cvd_high"].iloc[p2], color, dash, 1.5, 2)

            # Live signal — already detected on UTC data before timezone conversion
            for data in (low_data, high_data):
                if not data:
                    continue
                color = "cyan" if "SELLER" in data["signal"] else "magenta"
                dash  = "dash" if "EXHAUSTION" in data["signal"] else "solid"
                is_low = "SELLER" in data["signal"]
                p_col, c_col = ("p_low", "cvd_low") if is_low else ("p_high", "cvd_high")
                pivots = p_lows if is_low else p_highs
                if pivots:
                    last_p = pivots[-1]
                    t0 = merged["timestamp"].iloc[last_p]
                    t1 = merged["timestamp"].iloc[curr]
                    draw_line(t0, merged[p_col].iloc[last_p], t1, merged[p_col].iloc[curr], color, dash, 3, 1)
                    draw_line(t0, merged[c_col].iloc[last_p], t1, merged[c_col].iloc[curr], color, dash, 3, 2)

                p_pct = data.get("price_move_pct")
                c_pct = data.get("cvd_move_pct")
                score = data.get("div_score")
                metrics = ""
                if p_pct is not None and c_pct is not None:
                    score_str = f"  Δ{score:.1f}%" if score is not None else ""
                    metrics = f"  ·  price {p_pct:+.2f}%  CVD {c_pct:+.2f}%{score_str}"
                active_signals.append((color, f"{data['signal']}{metrics}"))

    # Banner — placed ABOVE the chart using paper coordinates > 1
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
            xanchor="center", yanchor="bottom",
        )

    # Panel labels — positioned at top edge of each panel, in the gap between panels
    panel_labels = [
        (1.0,   "BTC/USDT  ·  Binance Spot"),
        (0.604, "CVD Spot  ·  Aggregated"),
        (0.396, "CVD Futures  ·  Aggregated"),
        (0.188, "Open Interest  ·  Binance Futures"),
    ]
    for y_paper, text in panel_labels:
        fig.add_annotation(
            x=0.005, y=y_paper, xref="paper", yref="paper",
            text=f"<b>{text}</b>", showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,0.45)"),
            align="left", xanchor="left", yanchor="bottom",
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
        margin=dict(l=0, r=55, t=40, b=25),
        showlegend=False,
        hovermode=False,
    )

    return fig
