"""
app.py — Dash application for aggregated CVD scanner.

Reads from:
  data/btc_spot_5m.parquet     (spot klines, per exchange)
  data/btc_futures_5m.parquet  (futures klines, per exchange)
  data/btc_oi_1m.parquet       (OI snapshots, 1m granularity)

Panels:
  1. BTC/USDT Price        — Binance spot as reference
  2. CVD Spot Aggregated   — toggle per exchange
  3. CVD Futures Aggregated — toggle per exchange
  4. Open Interest          — Binance Futures

Run:   python app.py
Open:  http://localhost:8050
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")

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

# pandas resample strings for each UI interval option
INTERVAL_MAP = {
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
}

REFRESH_MS = 10_000      # 10 seconds
DISPLAY_CANDLES = 200   # candles shown per timeframe (same visual density on all TFs)

# Candle duration per interval (seconds) — used for x-axis right padding
INTERVAL_SECONDS = {
    "5m":  5 * 60,
    "15m": 15 * 60,
    "30m": 30 * 60,
    "1h":  60 * 60,
}

# Nice tick intervals in milliseconds — used when picking closest to target spacing
_NICE_TICK_MS = [h * 3_600_000 for h in (1, 2, 3, 4, 6, 8, 12, 24, 48)]

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_parquet(filepath: Path) -> pd.DataFrame:
    """Load parquet file. Returns empty DataFrame if file not found."""
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
      4. CVD_Open = CVD_Close shifted by 1 (so candle opens where previous closed)
      5. CVD_High/Low = max/min of Open and Close (intrabar moves not available via klines)
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
    """
    Resample 1m OI snapshots into OHLC candles for the target interval.
    Each candle represents the range of open interest within that period.
    """
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
# DIVERGENCE DETECTION
# ---------------------------------------------------------------------------

PIVOT_WINDOW = 5  # candles on each side required to confirm a pivot


def find_pivot_indices(series: pd.Series, window: int = PIVOT_WINDOW) -> Tuple[list, list]:
    """
    Find pivot lows and highs in a price/indicator Series.

    A pivot LOW  at index i: series[i] is the minimum within [i-window, i+window]
    A pivot HIGH at index i: series[i] is the maximum within [i-window, i+window]

    Returns (low_indices, high_indices) as lists of integer positional indices.
    Note: first and last `window` candles cannot be pivots by definition.
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


def add_divergences(
    fig:         go.Figure,
    price_df:    pd.DataFrame,
    ind_df:      pd.DataFrame,
    ind_high_col: str,
    ind_low_col:  str,
    price_row:   int,
    ind_row:     int,
) -> Tuple[str, str]:
    """
    Detect divergences between price and an indicator (CVD or OI).
    Draws historical divergence lines on both the price panel and the indicator panel.
    Returns (active_low_signal, active_high_signal) for the live candle.

    Divergence types (lows — bullish signals, cyan):
      Regular:  price lower low  + indicator higher low  → SELLERS EXHAUSTION    (dashed)
      Hidden:   price higher low + indicator lower low   → SELLERS ABSORPTION    (solid)

    Divergence types (highs — bearish signals, magenta):
      Regular:  price higher high + indicator lower high → BUYERS EXHAUSTION     (dashed)
      Hidden:   price lower high  + indicator higher high → BUYERS ABSORPTION    (solid)
    """
    if price_df.empty or ind_df.empty:
        return "", ""

    # Align price and indicator on the same timestamps (inner join)
    merged = pd.merge(
        price_df[["timestamp", "high", "low"]].rename(columns={"high": "p_high", "low": "p_low"}),
        ind_df[["timestamp", ind_high_col, ind_low_col]],
        on="timestamp", how="inner",
    ).reset_index(drop=True)

    if len(merged) < PIVOT_WINDOW * 2 + 2:
        return "", ""

    p_lows, p_highs = find_pivot_indices(merged["p_low"], PIVOT_WINDOW)
    curr = len(merged) - 1

    def draw_line(x0, y0, x1, y1, color, dash, width, row):
        fig.add_shape(
            type="line", x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color=color, width=width, dash=dash),
            row=row, col=1,
        )

    # ── Historical divergences on LOWS (bullish) ─────────────────────────
    for i in range(1, len(p_lows)):
        p1, p2 = p_lows[i - 1], p_lows[i]
        pdif = merged["p_low"].iloc[p2]          - merged["p_low"].iloc[p1]
        cdif = merged[ind_low_col].iloc[p2]      - merged[ind_low_col].iloc[p1]

        if   pdif < 0 and cdif > 0:  color, dash = "cyan", "dash"   # regular bullish
        elif pdif > 0 and cdif < 0:  color, dash = "cyan", "solid"  # hidden bullish
        else: continue

        t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
        draw_line(t0, merged["p_low"].iloc[p1],      t1, merged["p_low"].iloc[p2],      color, dash, 1.5, price_row)
        draw_line(t0, merged[ind_low_col].iloc[p1],  t1, merged[ind_low_col].iloc[p2],  color, dash, 1.5, ind_row)

    # ── Historical divergences on HIGHS (bearish) ────────────────────────
    for i in range(1, len(p_highs)):
        p1, p2 = p_highs[i - 1], p_highs[i]
        pdif = merged["p_high"].iloc[p2]         - merged["p_high"].iloc[p1]
        cdif = merged[ind_high_col].iloc[p2]     - merged[ind_high_col].iloc[p1]

        if   pdif > 0 and cdif < 0:  color, dash = "magenta", "dash"   # regular bearish
        elif pdif < 0 and cdif > 0:  color, dash = "magenta", "solid"  # hidden bearish
        else: continue

        t0, t1 = merged["timestamp"].iloc[p1], merged["timestamp"].iloc[p2]
        draw_line(t0, merged["p_high"].iloc[p1],     t1, merged["p_high"].iloc[p2],     color, dash, 1.5, price_row)
        draw_line(t0, merged[ind_high_col].iloc[p1], t1, merged[ind_high_col].iloc[p2], color, dash, 1.5, ind_row)

    # ── Live signal — current candle vs last confirmed pivot ─────────────
    active_low, active_high = "", ""

    # Check low: current candle is the lowest since last confirmed pivot low
    if p_lows:
        last_l = p_lows[-1]
        since_last = merged["p_low"].iloc[last_l + 1: curr + 1]
        if not since_last.empty and since_last.min() == merged["p_low"].iloc[curr]:
            l_pdif = merged["p_low"].iloc[curr]      - merged["p_low"].iloc[last_l]
            l_cdif = merged[ind_low_col].iloc[curr]  - merged[ind_low_col].iloc[last_l]
            if   l_pdif < 0 and l_cdif > 0: active_low = "SELLERS EXHAUSTION"
            elif l_pdif > 0 and l_cdif < 0: active_low = "SELLERS ABSORPTION"
            if active_low:
                t0, t1 = merged["timestamp"].iloc[last_l], merged["timestamp"].iloc[curr]
                lw = 3
                draw_line(t0, merged["p_low"].iloc[last_l],      t1, merged["p_low"].iloc[curr],      "cyan", "dash" if "EXHAUSTION" in active_low else "solid", lw, price_row)
                draw_line(t0, merged[ind_low_col].iloc[last_l],  t1, merged[ind_low_col].iloc[curr],  "cyan", "dash" if "EXHAUSTION" in active_low else "solid", lw, ind_row)

    # Check high: current candle is the highest since last confirmed pivot high
    if p_highs:
        last_h = p_highs[-1]
        since_last = merged["p_high"].iloc[last_h + 1: curr + 1]
        if not since_last.empty and since_last.max() == merged["p_high"].iloc[curr]:
            h_pdif = merged["p_high"].iloc[curr]     - merged["p_high"].iloc[last_h]
            h_cdif = merged[ind_high_col].iloc[curr] - merged[ind_high_col].iloc[last_h]
            if   h_pdif > 0 and h_cdif < 0: active_high = "BUYERS EXHAUSTION"
            elif h_pdif < 0 and h_cdif > 0: active_high = "BUYERS ABSORPTION"
            if active_high:
                t0, t1 = merged["timestamp"].iloc[last_h], merged["timestamp"].iloc[curr]
                lw = 3
                draw_line(t0, merged["p_high"].iloc[last_h],     t1, merged["p_high"].iloc[curr],     "magenta", "dash" if "EXHAUSTION" in active_high else "solid",   lw, price_row)
                draw_line(t0, merged[ind_high_col].iloc[last_h], t1, merged[ind_high_col].iloc[curr], "magenta", "dash" if "EXHAUSTION" in active_high else "solid",   lw, ind_row)

    return active_low, active_high


# ---------------------------------------------------------------------------
# FIGURE BUILDER
# ---------------------------------------------------------------------------

CANDLE_UP   = "#26a69a"
CANDLE_DOWN = "#ef5350"


def build_figure(
    price_df:       pd.DataFrame,
    cvd_spot_df:    pd.DataFrame,
    cvd_futures_df: pd.DataFrame,
    oi_df:          pd.DataFrame,
    show_divergences: bool = True,
    interval_str: str = "15m",
) -> go.Figure:
    """Assemble the 4-panel Plotly figure."""

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
    active_signals = []  # collect live signals from all panels

    if show_divergences:
        # CVD Spot (panel 2) vs Price (panel 1)
        low_sig, high_sig = add_divergences(
            fig, price_df, cvd_spot_df,
            ind_high_col="cvd_high", ind_low_col="cvd_low",
            price_row=1, ind_row=2,
        )
        for sig in (low_sig, high_sig):
            if sig:
                active_signals.append(("cyan" if "SELLER" in sig else "magenta", f"SPOT · {sig}"))

        # CVD Futures divergences disabled — futures data still accumulating


    # Live signal banner (shown only when at least one active signal exists)
    if active_signals:
        signal_text = "   |   ".join(f"<b>{label}</b>" for _, label in active_signals)
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

    # Panel labels (positioned in paper coordinates)
    # Boundaries with row_heights=[0.40,0.20,0.20,0.20], vertical_spacing=0.02:
    #   available_height = 1 - 3*0.02 = 0.94
    #   row4: [0,     0.188],  row3: [0.208, 0.396]
    #   row2: [0.416, 0.604],  row1: [0.624, 1.0  ]
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

    # Horizontal borders between panels — center of each gap in paper coords
    for border_y in [0.614, 0.406, 0.198]:
        fig.add_shape(
            type="line",
            x0=0, x1=1, y0=border_y, y1=border_y,
            xref="paper", yref="paper",
            line=dict(color="#593434", width=2),
        )

    # X-axis range: extend right by 10 empty candles for visual breathing room
    # Also compute dtick dynamically to target ~20 ticks across the visible range
    dtick = _NICE_TICK_MS[0]  # fallback: 1h
    if not price_df.empty:
        candle_td = pd.Timedelta(seconds=INTERVAL_SECONDS.get(interval_str, 900))
        x_min = price_df["timestamp"].iloc[0]
        x_max = price_df["timestamp"].iloc[-1] + candle_td * 10
        fig.update_xaxes(range=[x_min, x_max])
        total_ms = int((x_max - x_min).total_seconds() * 1000)
        target_ms = total_ms / 20
        dtick = min(_NICE_TICK_MS, key=lambda v: abs(v - target_ms))

    # Axes
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

    return fig


# ---------------------------------------------------------------------------
# DASH APP
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="CVD Scanner · BTC/USDT")

# Styles
_DARK   = "#111"
_PANEL  = "#161616"
_BORDER = "#2a2a2a"
_TEXT   = "#bbb"
_MUTED  = "rgba(255,255,255,0.35)"
_LABEL  = {
    "color": _MUTED,
    "fontSize": "10px",
    "textTransform": "uppercase",
    "letterSpacing": "0.8px",
    "marginRight": "8px",
    "whiteSpace": "nowrap",
}
_CHECKLIST_LABEL = {"color": _TEXT, "marginRight": "10px", "fontSize": "12px"}
_CHECKLIST_INPUT = {"marginRight": "4px", "accentColor": "#26a69a"}

app.layout = html.Div(
    style={"backgroundColor": _DARK, "height": "100vh", "display": "flex", "flexDirection": "column", "fontFamily": "system-ui, sans-serif"},
    children=[

        # ── Top controls bar ──────────────────────────────────────────────
        html.Div(
            style={
                "backgroundColor": _PANEL,
                "borderBottom": f"1px solid {_BORDER}",
                "padding": "6px 16px",
                "display": "flex",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "20px",
                "minHeight": "40px",
            },
            children=[

                # Symbol label
                html.Span("BTC/USDT", style={"color": "white", "fontWeight": "bold", "fontSize": "13px", "marginRight": "4px"}),

                # Interval selector
                html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                    html.Span("Interval", style=_LABEL),
                    dcc.RadioItems(
                        id="interval-selector",
                        options=[{"label": v, "value": v} for v in ["5m", "15m", "30m", "1h"]],
                        value="15m",
                        inline=True,
                        inputStyle=_CHECKLIST_INPUT,
                        labelStyle=_CHECKLIST_LABEL,
                    ),
                ]),

                # Spot exchange toggles
                html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                    html.Span("CVD Spot", style=_LABEL),
                    dcc.Checklist(
                        id="spot-exchanges",
                        options=[{"label": label, "value": key} for key, label in SPOT_EXCHANGES.items()],
                        value=list(SPOT_EXCHANGES.keys()),
                        inline=True,
                        inputStyle=_CHECKLIST_INPUT,
                        labelStyle=_CHECKLIST_LABEL,
                    ),
                ]),

                # Futures exchange toggles
                html.Div(style={"display": "flex", "alignItems": "center"}, children=[
                    html.Span("CVD Futures", style=_LABEL),
                    dcc.Checklist(
                        id="futures-exchanges",
                        options=[{"label": label, "value": key} for key, label in FUTURES_EXCHANGES.items()],
                        value=list(FUTURES_EXCHANGES.keys()),
                        inline=True,
                        inputStyle=_CHECKLIST_INPUT,
                        labelStyle=_CHECKLIST_LABEL,
                    ),
                ]),

                # Last updated timestamp (right-aligned)
                html.Div(
                    id="last-updated",
                    style={"color": _MUTED, "fontSize": "11px", "marginLeft": "auto"},
                ),
            ],
        ),

        # ── Chart ─────────────────────────────────────────────────────────
        dcc.Graph(
            id="main-chart",
            style={"flex": "1", "minHeight": 0},
            config={"responsive": True, "displayModeBar": False},
        ),

        # ── Auto-refresh ticker ───────────────────────────────────────────
        dcc.Interval(id="refresh-interval", interval=REFRESH_MS, n_intervals=0),
    ],
)


# ---------------------------------------------------------------------------
# CALLBACK
# ---------------------------------------------------------------------------

@app.callback(
    Output("main-chart", "figure"),
    Output("last-updated", "children"),
    Input("refresh-interval", "n_intervals"),
    Input("interval-selector", "value"),
    Input("spot-exchanges", "value"),
    Input("futures-exchanges", "value"),
)
def update_chart(_, interval_str, spot_selected, futures_selected):
    """
    Triggered every 10s (auto-refresh) or on any control change.
    Reloads data from disk, resamples, computes CVD/OI, rebuilds figure.
    """
    pandas_interval = INTERVAL_MAP[interval_str]

    # Load raw data from disk
    spot_raw    = load_parquet(DATA_DIR / "btc_spot_5m.parquet")
    futures_raw = load_parquet(DATA_DIR / "btc_futures_5m.parquet")
    oi_raw      = load_parquet(DATA_DIR / "btc_oi_1m.parquet")

    # Resample klines to selected interval
    spot_rs    = resample_klines(spot_raw,    pandas_interval)
    futures_rs = resample_klines(futures_raw, pandas_interval)

    # Build datasets for each panel, trimmed to last DISPLAY_CANDLES
    price_df       = trim_to_candles(get_price_df(spot_rs),                         DISPLAY_CANDLES)
    cvd_spot_df    = trim_to_candles(compute_cvd(spot_rs,    spot_selected    or []), DISPLAY_CANDLES)
    cvd_futures_df = trim_to_candles(compute_cvd(futures_rs, futures_selected or []), DISPLAY_CANDLES)
    oi_df          = trim_to_candles(compute_oi_ohlc(oi_raw, pandas_interval),       DISPLAY_CANDLES)

    fig = build_figure(price_df, cvd_spot_df, cvd_futures_df, oi_df, interval_str=interval_str)
    now_str = "Updated " + datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

    return fig, now_str


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # host="0.0.0.0" makes the app accessible from other machines (VPS, LAN)
    app.run(debug=False, host="0.0.0.0", port=8050)
