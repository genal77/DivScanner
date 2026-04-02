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

Run:   python3 app.py
Open:  http://localhost:8050
"""

import os
import pandas as pd
import dash
import dash_auth
from dash import dcc, html, Input, Output
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

_WARSAW = ZoneInfo("Europe/Warsaw")
from dotenv import load_dotenv

load_dotenv()

from analysis import (
    INTERVAL_MAP, DISPLAY_CANDLES, PIVOT_WINDOW,
    SPOT_EXCHANGES, FUTURES_EXCHANGES,
    load_parquet, resample_klines, compute_cvd, compute_oi_ohlc,
    to_warsaw,
    trim_to_candles, get_price_df, build_figure,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DATA_DIR   = Path("data")
REFRESH_MS = 10_000

# ---------------------------------------------------------------------------
# DASH APP
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="CVD Scanner · BTC/USDT")
dash_auth.BasicAuth(app, {os.getenv("DASH_USER", "admin"): os.getenv("DASH_PASSWORD", "")})

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
_SLIDER_STYLE    = {"width": "120px", "display": "inline-block", "verticalAlign": "middle"}
_SLIDER_CSS      = {"height": "18px"}

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
                html.Span("BTC/USDT", style={"color": "white", "fontWeight": "bold", "fontSize": "13px", "marginRight": "4px"}),

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

                # ── Pivot controls ────────────────────────────────────────
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px"}, children=[
                    html.Span("Pivots", style=_LABEL),
                    dcc.Checklist(
                        id="show-pivots",
                        options=[{"label": "", "value": "show"}],
                        value=[],
                        inputStyle=_CHECKLIST_INPUT,
                        labelStyle=_CHECKLIST_LABEL,
                    ),
                    html.Span("LB", style={**_LABEL, "marginRight": "4px"}),
                    html.Div(style=_SLIDER_STYLE, children=[
                        dcc.Slider(
                            id="pivot-lb",
                            min=1, max=10, step=1, value=PIVOT_WINDOW,
                            marks={i: {"label": str(i), "style": {"color": _MUTED, "fontSize": "9px"}} for i in range(1, 11)},
                            tooltip={"placement": "top", "always_visible": False},
                            className="pivot-slider",
                        ),
                    ]),
                    html.Span("RB", style={**_LABEL, "marginLeft": "8px", "marginRight": "4px"}),
                    html.Div(style=_SLIDER_STYLE, children=[
                        dcc.Slider(
                            id="pivot-rb",
                            min=1, max=10, step=1, value=PIVOT_WINDOW,
                            marks={i: {"label": str(i), "style": {"color": _MUTED, "fontSize": "9px"}} for i in range(1, 11)},
                            tooltip={"placement": "top", "always_visible": False},
                            className="pivot-slider",
                        ),
                    ]),
                ]),

                html.Div(id="last-updated", style={"color": _MUTED, "fontSize": "11px", "marginLeft": "auto"}),
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
    Input("show-pivots", "value"),
    Input("pivot-lb", "value"),
    Input("pivot-rb", "value"),
)
def update_chart(_, interval_str, spot_selected, futures_selected, show_pivots_val, pivot_lb, pivot_rb):
    """
    Triggered every 10s (auto-refresh) or on any control change.
    Reloads data from disk, resamples, computes CVD/OI, rebuilds figure.
    """
    pandas_interval = INTERVAL_MAP[interval_str]

    spot_raw    = load_parquet(DATA_DIR / "btc_spot_5m.parquet")
    futures_raw = load_parquet(DATA_DIR / "btc_futures_5m.parquet")
    oi_raw      = load_parquet(DATA_DIR / "btc_oi_1m.parquet")

    spot_rs    = resample_klines(spot_raw,    pandas_interval)
    futures_rs = resample_klines(futures_raw, pandas_interval)

    price_df       = trim_to_candles(get_price_df(spot_rs),                          DISPLAY_CANDLES)
    cvd_spot_df    = trim_to_candles(compute_cvd(spot_rs,    spot_selected     or []), DISPLAY_CANDLES)
    cvd_futures_df = trim_to_candles(compute_cvd(futures_rs, futures_selected  or []), DISPLAY_CANDLES)
    oi_df          = trim_to_candles(compute_oi_ohlc(oi_raw, pandas_interval),        DISPLAY_CANDLES)

    price_df       = to_warsaw(price_df)
    cvd_spot_df    = to_warsaw(cvd_spot_df)
    cvd_futures_df = to_warsaw(cvd_futures_df)
    oi_df          = to_warsaw(oi_df)

    fig, _ = build_figure(
        price_df, cvd_spot_df, cvd_futures_df, oi_df,
        interval_str=interval_str,
        show_pivots=bool(show_pivots_val),
        pivot_left=pivot_lb or PIVOT_WINDOW,
        pivot_right=pivot_rb or PIVOT_WINDOW,
    )
    now_str = "Updated " + datetime.now(_WARSAW).strftime("%H:%M:%S (Warsaw)")

    return fig, now_str


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
