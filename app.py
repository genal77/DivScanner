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
from dash import dcc, html, Input, Output, State
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
# STYLES
# ---------------------------------------------------------------------------

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
    "marginRight": "6px",
    "whiteSpace": "nowrap",
}
_CHECKLIST_LABEL = {"color": _TEXT, "marginRight": "10px", "fontSize": "12px"}
_CHECKLIST_INPUT = {"marginRight": "4px", "accentColor": "#26a69a"}
_SLIDER_STYLE    = {"width": "110px", "display": "inline-block", "verticalAlign": "middle"}
_VAL_BADGE       = {
    "display": "inline-block",
    "width": "18px",
    "textAlign": "center",
    "color": "white",
    "fontSize": "11px",
    "fontWeight": "bold",
    "marginRight": "6px",
}

# Sidebar checklist labels — vertical layout for sidebar
_SIDEBAR_LABEL = {"color": _TEXT, "fontSize": "12px", "display": "block", "marginBottom": "5px"}

# ---------------------------------------------------------------------------
# DASH APP
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="CVD Scanner · BTC/USDT")
dash_auth.BasicAuth(app, {os.getenv("DASH_USER", "admin"): os.getenv("DASH_PASSWORD", "")})

app.layout = html.Div(
    style={
        "backgroundColor": _DARK,
        "height": "100vh",
        "display": "flex",
        "flexDirection": "column",
        "fontFamily": "system-ui, sans-serif",
        "overflow": "hidden",
    },
    children=[

        # ── Persistent state stores ───────────────────────────────────────
        dcc.Store(
            id="exchange-state",
            storage_type="local",
            data={
                "spot":    list(SPOT_EXCHANGES.keys()),
                "futures": list(FUTURES_EXCHANGES.keys()),
            },
        ),

        # ── Top controls bar ──────────────────────────────────────────────
        html.Div(
            style={
                "backgroundColor": _PANEL,
                "borderBottom": f"1px solid {_BORDER}",
                "padding": "6px 16px",
                "display": "flex",
                "alignItems": "center",
                "flexWrap": "nowrap",
                "gap": "20px",
                "minHeight": "40px",
                "flexShrink": "0",
            },
            children=[
                html.Span(
                    "BTC/USDT",
                    style={"color": "white", "fontWeight": "bold", "fontSize": "13px"},
                ),

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

                # Pivot controls
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "6px"}, children=[
                    html.Span("Pivots", style=_LABEL),
                    dcc.Checklist(
                        id="show-pivots",
                        options=[{"label": "", "value": "show"}],
                        value=[],
                        inputStyle=_CHECKLIST_INPUT,
                        labelStyle=_CHECKLIST_LABEL,
                    ),
                    html.Span("LB", style={**_LABEL, "marginRight": "4px"}),
                    html.Span(str(PIVOT_WINDOW), id="pivot-lb-val", style=_VAL_BADGE),
                    html.Div(style=_SLIDER_STYLE, children=[
                        dcc.Slider(
                            id="pivot-lb",
                            min=1, max=10, step=1, value=PIVOT_WINDOW,
                            marks={i: {"label": str(i), "style": {"color": _MUTED, "fontSize": "9px"}} for i in range(1, 11)},
                            tooltip={"placement": "top", "always_visible": False},
                            className="pivot-slider",
                        ),
                    ]),
                    html.Span("RB", style={**_LABEL, "marginLeft": "6px", "marginRight": "4px"}),
                    html.Span(str(PIVOT_WINDOW), id="pivot-rb-val", style=_VAL_BADGE),
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

                # Last updated timestamp
                html.Div(id="last-updated", style={"color": _MUTED, "fontSize": "11px", "marginLeft": "auto"}),
            ],
        ),

        # ── Main area: chart + sidebar ─────────────────────────────────────
        html.Div(
            style={"display": "flex", "flex": "1", "minHeight": "0"},
            children=[

                # Chart
                dcc.Graph(
                    id="main-chart",
                    style={"flex": "1", "minHeight": "0"},
                    config={"responsive": True, "displayModeBar": False},
                ),

                # Right sidebar
                html.Div(
                    id="sidebar",
                    style={
                        "backgroundColor": _PANEL,
                        "borderLeft": f"1px solid {_BORDER}",
                        "display": "flex",
                        "flexDirection": "column",
                        "flexShrink": "0",
                        "width": "200px",
                        "transition": "width 0.15s ease",
                    },
                    children=[
                        # Toggle button row
                        html.Div(
                            style={
                                "padding": "8px 8px 4px 8px",
                                "display": "flex",
                                "justifyContent": "flex-end",
                            },
                            children=[
                                html.Button(
                                    "▶",
                                    id="sidebar-toggle",
                                    n_clicks=0,
                                    style={
                                        "background": "none",
                                        "border": f"1px solid {_BORDER}",
                                        "color": _MUTED,
                                        "cursor": "pointer",
                                        "fontSize": "10px",
                                        "padding": "2px 6px",
                                        "borderRadius": "3px",
                                    },
                                ),
                            ],
                        ),

                        # Sidebar content (exchange toggles)
                        html.Div(
                            id="sidebar-content",
                            style={"padding": "4px 14px 14px 14px", "overflowY": "auto"},
                            children=[

                                # CVD Spot section
                                html.Div(
                                    style={"marginBottom": "18px"},
                                    children=[
                                        html.Div(
                                            "CVD Spot",
                                            style={
                                                "color": _MUTED,
                                                "fontSize": "10px",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.8px",
                                                "marginBottom": "8px",
                                                "paddingBottom": "4px",
                                                "borderBottom": f"1px solid {_BORDER}",
                                            },
                                        ),
                                        dcc.Checklist(
                                            id="spot-exchanges",
                                            options=[{"label": label, "value": key} for key, label in SPOT_EXCHANGES.items()],
                                            value=list(SPOT_EXCHANGES.keys()),
                                            inputStyle=_CHECKLIST_INPUT,
                                            labelStyle=_SIDEBAR_LABEL,
                                        ),
                                    ],
                                ),

                                # CVD Futures section
                                html.Div(
                                    children=[
                                        html.Div(
                                            "CVD Futures",
                                            style={
                                                "color": _MUTED,
                                                "fontSize": "10px",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "0.8px",
                                                "marginBottom": "8px",
                                                "paddingBottom": "4px",
                                                "borderBottom": f"1px solid {_BORDER}",
                                            },
                                        ),
                                        dcc.Checklist(
                                            id="futures-exchanges",
                                            options=[{"label": label, "value": key} for key, label in FUTURES_EXCHANGES.items()],
                                            value=list(FUTURES_EXCHANGES.keys()),
                                            inputStyle=_CHECKLIST_INPUT,
                                            labelStyle=_SIDEBAR_LABEL,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # ── Auto-refresh ticker ───────────────────────────────────────────
        dcc.Interval(id="refresh-interval", interval=REFRESH_MS, n_intervals=0),
    ],
)

# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------

@app.callback(
    Output("spot-exchanges", "value"),
    Output("futures-exchanges", "value"),
    Input("exchange-state", "data"),
    prevent_initial_call=False,
)
def restore_exchange_state(state):
    """On page load, restore exchange selections from localStorage."""
    if not state:
        return list(SPOT_EXCHANGES.keys()), list(FUTURES_EXCHANGES.keys())
    return (
        state.get("spot",    list(SPOT_EXCHANGES.keys())),
        state.get("futures", list(FUTURES_EXCHANGES.keys())),
    )


@app.callback(
    Output("exchange-state", "data"),
    Input("spot-exchanges", "value"),
    Input("futures-exchanges", "value"),
    prevent_initial_call=True,
)
def save_exchange_state(spot, futures):
    """Persist exchange selections to localStorage on every change."""
    return {"spot": spot or [], "futures": futures or []}


@app.callback(
    Output("sidebar-content", "style"),
    Output("sidebar", "style"),
    Output("sidebar-toggle", "children"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-toggle", "children"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, current_label):
    """Toggle sidebar open/closed."""
    if current_label == "▶":
        # Currently collapsed → expand
        return (
            {"padding": "4px 14px 14px 14px", "overflowY": "auto"},
            {
                "backgroundColor": _PANEL,
                "borderLeft": f"1px solid {_BORDER}",
                "display": "flex",
                "flexDirection": "column",
                "flexShrink": "0",
                "width": "200px",
                "transition": "width 0.15s ease",
            },
            "▶",
        )
    else:
        # Currently expanded → collapse
        return (
            {"display": "none"},
            {
                "backgroundColor": _PANEL,
                "borderLeft": f"1px solid {_BORDER}",
                "display": "flex",
                "flexDirection": "column",
                "flexShrink": "0",
                "width": "32px",
                "transition": "width 0.15s ease",
            },
            "◀",
        )


@app.callback(
    Output("main-chart", "figure"),
    Output("last-updated", "children"),
    Output("pivot-lb-val", "children"),
    Output("pivot-rb-val", "children"),
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

    lb = pivot_lb or PIVOT_WINDOW
    rb = pivot_rb or PIVOT_WINDOW

    fig, _ = build_figure(
        price_df, cvd_spot_df, cvd_futures_df, oi_df,
        interval_str=interval_str,
        show_pivots=bool(show_pivots_val),
        pivot_left=lb,
        pivot_right=rb,
    )
    now_str = "Updated " + datetime.now(_WARSAW).strftime("%H:%M:%S (Warsaw)")

    return fig, now_str, str(lb), str(rb)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
