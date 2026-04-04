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
    trim_to_candles, reset_cvd_origin, get_price_df, build_figure,
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

# Sidebar checklist labels — vertical layout
_SIDEBAR_LABEL = {
    "color": _TEXT,
    "fontSize": "12px",
    "display": "block",
    "marginBottom": "5px",
    "whiteSpace": "nowrap",
}

_SECTION_HEADER = {
    "color": _MUTED,
    "fontSize": "10px",
    "textTransform": "uppercase",
    "letterSpacing": "0.8px",
    "marginBottom": "8px",
    "paddingBottom": "4px",
    "borderBottom": f"1px solid {_BORDER}",
    "whiteSpace": "nowrap",
}

# Sidebar open state style
_SIDEBAR_OPEN = {
    "backgroundColor": _PANEL,
    "borderLeft": f"1px solid {_BORDER}",
    "display": "flex",
    "flexDirection": "column",
    "flexShrink": "0",
    "width": "fit-content",
}

# Sidebar collapsed state style
_SIDEBAR_COLLAPSED = {
    "backgroundColor": _PANEL,
    "borderLeft": f"1px solid {_BORDER}",
    "display": "flex",
    "flexDirection": "column",
    "flexShrink": "0",
    "width": "32px",
}

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
        dcc.Store(
            id="chart-modes",
            storage_type="local",
            data={"cvd_spot": "candle", "cvd_futures": "candle", "oi": "candle"},
        ),

        # ── Tab nav ───────────────────────────────────────────────────────
        html.Div(
            style={
                "backgroundColor": _PANEL,
                "borderBottom": f"1px solid {_BORDER}",
                "display": "flex",
                "flexShrink": "0",
            },
            children=[
                html.Button("Chart",      id="tab-btn-chart",      n_clicks=0,
                    style={"background":"none","border":"none","borderBottom":"2px solid #26a69a",
                           "color":"white","padding":"8px 20px","cursor":"pointer","fontSize":"12px","fontFamily":"system-ui,sans-serif"}),
                html.Button("Signal Log", id="tab-btn-signal-log", n_clicks=0,
                    style={"background":"none","border":"none","borderBottom":"2px solid transparent",
                           "color":_MUTED,"padding":"8px 20px","cursor":"pointer","fontSize":"12px","fontFamily":"system-ui,sans-serif"}),
            ],
        ),

        # ── Chart area ────────────────────────────────────────────────────
        html.Div(
            id="chart-area",
            style={"display": "flex", "flexDirection": "column", "flex": "1", "minHeight": "0"},
            children=[

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

                # Last updated timestamp
                html.Div(id="last-updated", style={"color": _MUTED, "fontSize": "11px", "marginLeft": "auto"}),
            ],
        ),

        # ── Main area: chart + sidebar ────────────────────────────────────
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
                    style=_SIDEBAR_OPEN,
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

                        # Sidebar content
                        html.Div(
                            id="sidebar-content",
                            style={"padding": "4px 14px 14px 14px", "overflowY": "auto"},
                            children=[

                                # CVD Spot section
                                html.Div(
                                    style={"marginBottom": "18px"},
                                    children=[
                                        html.Div("CVD Spot", style=_SECTION_HEADER),
                                        dcc.Checklist(
                                            id="spot-exchanges",
                                            options=[{"label": label, "value": key} for key, label in SPOT_EXCHANGES.items()],
                                            value=list(SPOT_EXCHANGES.keys()),
                                            inputStyle=_CHECKLIST_INPUT,
                                            labelStyle=_SIDEBAR_LABEL,
                                        ),
                                        dcc.RadioItems(
                                            id="cvd-spot-mode",
                                            options=[{"label": "Candle", "value": "candle"}, {"label": "Line", "value": "line"}],
                                            value="candle",
                                            inline=True,
                                            inputStyle={**_CHECKLIST_INPUT, "marginTop": "8px"},
                                            labelStyle={**_CHECKLIST_LABEL, "fontSize": "11px"},
                                        ),
                                    ],
                                ),

                                # CVD Futures section
                                html.Div(
                                    style={"marginBottom": "18px"},
                                    children=[
                                        html.Div("CVD Futures", style=_SECTION_HEADER),
                                        dcc.Checklist(
                                            id="futures-exchanges",
                                            options=[{"label": label, "value": key} for key, label in FUTURES_EXCHANGES.items()],
                                            value=list(FUTURES_EXCHANGES.keys()),
                                            inputStyle=_CHECKLIST_INPUT,
                                            labelStyle=_SIDEBAR_LABEL,
                                        ),
                                        dcc.RadioItems(
                                            id="cvd-futures-mode",
                                            options=[{"label": "Candle", "value": "candle"}, {"label": "Line", "value": "line"}],
                                            value="candle",
                                            inline=True,
                                            inputStyle={**_CHECKLIST_INPUT, "marginTop": "8px"},
                                            labelStyle={**_CHECKLIST_LABEL, "fontSize": "11px"},
                                        ),
                                    ],
                                ),

                                # Open Interest section
                                html.Div(
                                    style={"marginBottom": "18px"},
                                    children=[
                                        html.Div("Open Interest", style=_SECTION_HEADER),
                                        dcc.RadioItems(
                                            id="oi-mode",
                                            options=[{"label": "Candle", "value": "candle"}, {"label": "Line", "value": "line"}],
                                            value="candle",
                                            inline=True,
                                            inputStyle=_CHECKLIST_INPUT,
                                            labelStyle={**_CHECKLIST_LABEL, "fontSize": "11px"},
                                        ),
                                    ],
                                ),

                                # Pivots section
                                html.Div(
                                    children=[
                                        html.Div("Pivots", style=_SECTION_HEADER),
                                        dcc.Checklist(
                                            id="show-pivots",
                                            options=[{"label": "Show", "value": "show"}],
                                            value=[],
                                            inputStyle=_CHECKLIST_INPUT,
                                            labelStyle={**_SIDEBAR_LABEL, "marginBottom": "10px"},
                                        ),
                                        html.Div("LB", style={**_LABEL, "marginBottom": "4px"}),
                                        html.Div(style={"width": "100%", "marginBottom": "14px"}, children=[
                                            dcc.Slider(
                                                id="pivot-lb",
                                                min=1, max=15, step=1, value=PIVOT_WINDOW,
                                                marks={
                                                    1:  {"label": "1",  "style": {"color": _MUTED, "fontSize": "9px"}},
                                                    5:  {"label": "5",  "style": {"color": _MUTED, "fontSize": "9px"}},
                                                    10: {"label": "10", "style": {"color": _MUTED, "fontSize": "9px"}},
                                                    15: {"label": "15", "style": {"color": _MUTED, "fontSize": "9px"}},
                                                },
                                                tooltip={"placement": "top", "always_visible": False},
                                                className="pivot-slider",
                                            ),
                                        ]),
                                        html.Div("RB", style={**_LABEL, "marginBottom": "4px"}),
                                        html.Div(style={"width": "100%"}, children=[
                                            dcc.Slider(
                                                id="pivot-rb",
                                                min=1, max=15, step=1, value=PIVOT_WINDOW,
                                                marks={
                                                    1:  {"label": "1",  "style": {"color": _MUTED, "fontSize": "9px"}},
                                                    5:  {"label": "5",  "style": {"color": _MUTED, "fontSize": "9px"}},
                                                    10: {"label": "10", "style": {"color": _MUTED, "fontSize": "9px"}},
                                                    15: {"label": "15", "style": {"color": _MUTED, "fontSize": "9px"}},
                                                },
                                                tooltip={"placement": "top", "always_visible": False},
                                                className="pivot-slider",
                                            ),
                                        ]),
                                    ],
                                ),

                            ],
                        ),
                    ],
                ),
            ],
        ),

            ],
        ),  # end chart-area

        # ── Signal log area ───────────────────────────────────────────────
        html.Div(
            id="signal-log-area",
            style={"display": "none", "flexDirection": "column", "flex": "1", "minHeight": "0", "overflowY": "auto", "padding": "16px", "backgroundColor": _DARK},
            children=[
                html.Div(id="signal-log-meta", style={"color": _MUTED, "fontSize": "11px", "marginBottom": "10px"}),
                dash.dash_table.DataTable(
                    id="signal-log-table",
                    columns=[],
                    data=[],
                    sort_action="native",
                    page_size=50,
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": _PANEL, "color": _MUTED, "fontSize": "11px",
                        "fontWeight": "normal", "textTransform": "uppercase", "letterSpacing": "0.5px",
                        "borderBottom": f"1px solid {_BORDER}", "border": "none", "whiteSpace": "nowrap",
                    },
                    style_cell={
                        "backgroundColor": _DARK, "color": _TEXT, "fontSize": "12px",
                        "border": "none", "borderBottom": f"1px solid {_BORDER}",
                        "padding": "6px 10px", "whiteSpace": "nowrap", "fontFamily": "monospace",
                    },
                    style_data_conditional=[
                        {"if": {"filter_query": '{signal} contains "BUYING"'},  "color": "#ef5350"},
                        {"if": {"filter_query": '{signal} contains "SELLING"'}, "color": "#26a69a"},
                    ],
                ),
            ],
        ),  # end signal-log-area

        # ── Auto-refresh ticker ───────────────────────────────────────────
        dcc.Interval(id="refresh-interval", interval=REFRESH_MS, n_intervals=0),
        dcc.Interval(id="log-refresh-interval", interval=30_000, n_intervals=0),
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
    Output("cvd-spot-mode",    "value"),
    Output("cvd-futures-mode", "value"),
    Output("oi-mode",          "value"),
    Input("chart-modes", "data"),
    prevent_initial_call=False,
)
def restore_chart_modes(data):
    """On page load, restore chart mode selections from localStorage."""
    if not data:
        return "candle", "candle", "candle"
    return data.get("cvd_spot", "candle"), data.get("cvd_futures", "candle"), data.get("oi", "candle")


@app.callback(
    Output("chart-modes", "data"),
    Input("cvd-spot-mode",    "value"),
    Input("cvd-futures-mode", "value"),
    Input("oi-mode",          "value"),
    prevent_initial_call=True,
)
def save_chart_modes(cvd_spot, cvd_futures, oi):
    """Persist chart mode selections to localStorage on every change."""
    return {"cvd_spot": cvd_spot, "cvd_futures": cvd_futures, "oi": oi}


@app.callback(
    Output("sidebar-content", "style"),
    Output("sidebar", "style"),
    Output("sidebar-toggle", "children"),
    Input("sidebar-toggle", "n_clicks"),
    State("sidebar-toggle", "children"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, current_label):
    """Toggle sidebar open/closed. ▶ = open (click to collapse), ◀ = collapsed (click to expand)."""
    if current_label == "▶":
        # Currently open → collapse
        return {"display": "none"}, _SIDEBAR_COLLAPSED, "◀"
    else:
        # Currently collapsed → expand
        return {"padding": "4px 14px 14px 14px", "overflowY": "auto"}, _SIDEBAR_OPEN, "▶"


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
    Input("cvd-spot-mode",    "value"),
    Input("cvd-futures-mode", "value"),
    Input("oi-mode",          "value"),
)
def update_chart(_, interval_str, spot_selected, futures_selected, show_pivots_val,
                 pivot_lb, pivot_rb, cvd_spot_mode, cvd_futures_mode, oi_mode):
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

    price_df       = trim_to_candles(get_price_df(spot_rs),                           DISPLAY_CANDLES)
    cvd_spot_df    = reset_cvd_origin(trim_to_candles(compute_cvd(spot_rs,    spot_selected     or []), DISPLAY_CANDLES))
    cvd_futures_df = reset_cvd_origin(trim_to_candles(compute_cvd(futures_rs, futures_selected  or []), DISPLAY_CANDLES))
    oi_df          = trim_to_candles(compute_oi_ohlc(oi_raw, pandas_interval),         DISPLAY_CANDLES)

    price_df       = to_warsaw(price_df)
    cvd_spot_df    = to_warsaw(cvd_spot_df)
    cvd_futures_df = to_warsaw(cvd_futures_df)
    oi_df          = to_warsaw(oi_df)

    fig, _ = build_figure(
        price_df, cvd_spot_df, cvd_futures_df, oi_df,
        interval_str=interval_str,
        show_pivots=bool(show_pivots_val),
        pivot_left=pivot_lb  or PIVOT_WINDOW,
        pivot_right=pivot_rb or PIVOT_WINDOW,
        cvd_spot_mode=cvd_spot_mode    or "candle",
        cvd_futures_mode=cvd_futures_mode or "candle",
        oi_mode=oi_mode                or "candle",
    )
    now_str = "Updated " + datetime.now(_WARSAW).strftime("%H:%M:%S (Warsaw)")

    return fig, now_str


# ---------------------------------------------------------------------------
# SIGNAL LOG
# ---------------------------------------------------------------------------

_LOG_COLUMNS = [
    {"name": "Time (UTC+2)",   "id": "time_local"},
    {"name": "Signal",         "id": "signal"},
    {"name": "TF",             "id": "timeframes"},
    {"name": "Price →",        "id": "price_move"},
    {"name": "CVD →",          "id": "cvd_move"},
    {"name": "Persist.",       "id": "persistence_fmt"},
    {"name": "P.Move ATR",     "id": "price_atr_fmt"},
    {"name": "CVD σ",          "id": "cvd_sigma_fmt"},
    {"name": "Window",         "id": "window_bars"},
    {"name": "OI Δ%",          "id": "oi_delta_fmt"},
    {"name": "Fut.CVD Δ",      "id": "futures_cvd_fmt"},
    {"name": "BTC price",      "id": "btc_price"},
]


@app.callback(
    Output("chart-area",      "style"),
    Output("signal-log-area", "style"),
    Output("tab-btn-chart",      "style"),
    Output("tab-btn-signal-log", "style"),
    Input("tab-btn-chart",      "n_clicks"),
    Input("tab-btn-signal-log", "n_clicks"),
)
def switch_tab(n_chart, n_log):
    """Toggle chart / signal log visibility based on which tab button was clicked."""
    from dash import ctx
    active = "signal-log" if ctx.triggered_id == "tab-btn-signal-log" else "chart"
    _btn_active   = {"background":"none","border":"none","borderBottom":"2px solid #26a69a",
                     "color":"white","padding":"8px 20px","cursor":"pointer","fontSize":"12px","fontFamily":"system-ui,sans-serif"}
    _btn_inactive = {"background":"none","border":"none","borderBottom":"2px solid transparent",
                     "color":_MUTED,"padding":"8px 20px","cursor":"pointer","fontSize":"12px","fontFamily":"system-ui,sans-serif"}
    if active == "chart":
        return (
            {"display":"flex","flexDirection":"column","flex":"1","minHeight":"0"},
            {"display":"none"},
            _btn_active, _btn_inactive,
        )
    else:
        return (
            {"display":"none"},
            {"display":"flex","flexDirection":"column","flex":"1","minHeight":"0","overflowY":"auto","padding":"16px","backgroundColor":_DARK},
            _btn_inactive, _btn_active,
        )


@app.callback(
    Output("signal-log-table", "columns"),
    Output("signal-log-table", "data"),
    Output("signal-log-meta",  "children"),
    Input("log-refresh-interval", "n_intervals"),
    Input("tab-btn-signal-log",   "n_clicks"),
)
def update_signal_log(_, n_log):
    """Load signal_log.csv and format for DataTable. Triggered on tab switch and every 30s."""
    from dash import ctx
    if ctx.triggered_id == "log-refresh-interval" and (n_log or 0) == 0:
        return dash.no_update, dash.no_update, dash.no_update

    log_path = DATA_DIR / "signal_log.csv"
    if not log_path.exists():
        return _LOG_COLUMNS, [], "Brak danych — signal_log.csv nie istnieje."

    df = pd.read_csv(log_path)
    if df.empty:
        return _LOG_COLUMNS, [], "Plik istnieje, ale nie zawiera jeszcze żadnych sygnałów."

    df = df.sort_values("sent_at", ascending=False)

    _WA = ZoneInfo("Europe/Warsaw")

    def _fmt_ts(ts_str):
        try:
            return pd.Timestamp(ts_str).tz_convert(_WA).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ts_str

    def _fmt_price_move(row):
        try:
            return f"{row['price_from']:,.0f} → {row['price_to']:,.0f}"
        except Exception:
            return ""

    def _fmt_cvd_move(row):
        try:
            return f"{row['cvd_from']:+,.0f} → {row['cvd_to']:+,.0f}"
        except Exception:
            return ""

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "time_local":    _fmt_ts(r.get("sent_at", "")),
            "signal":        r.get("signal", ""),
            "timeframes":    r.get("timeframes", ""),
            "price_move":    _fmt_price_move(r),
            "cvd_move":      _fmt_cvd_move(r),
            "persistence_fmt": f"{r['persistence']*100:.0f}%" if pd.notna(r.get("persistence")) else "—",
            "price_atr_fmt": f"{r['price_atr_ratio']:.2f}×" if pd.notna(r.get("price_atr_ratio")) else "—",
            "cvd_sigma_fmt": f"{r['cvd_sigma']:.2f}σ"       if pd.notna(r.get("cvd_sigma"))       else "—",
            "window_bars":   int(r["window_bars"]) if pd.notna(r.get("window_bars")) else "—",
            "oi_delta_fmt":  f"{r['oi_delta_pct']:+.3f}%"   if pd.notna(r.get("oi_delta_pct"))    else "—",
            "futures_cvd_fmt": f"{r['futures_cvd_delta']:+,.0f}" if pd.notna(r.get("futures_cvd_delta")) else "—",
            "btc_price":     f"{r['btc_price']:,.2f}"        if pd.notna(r.get("btc_price"))        else "—",
        })

    meta = f"{len(df)} sygnałów · ostatni: {_fmt_ts(df['sent_at'].iloc[0])}"
    return _LOG_COLUMNS, rows, meta


# ---------------------------------------------------------------------------
# CROSSHAIR — clientside callback draws a full-height vertical line on hover
# ---------------------------------------------------------------------------

app.clientside_callback(
    """
    function(hoverData, figure) {
        if (!hoverData || !figure) return window.dash_clientside.no_update;

        var x = hoverData.points[0].x;
        var shapes = (figure.layout.shapes || []).filter(function(s) {
            return s.name !== 'crosshair';
        });
        shapes.push({
            name: 'crosshair',
            type: 'line',
            x0: x, x1: x,
            y0: 0, y1: 1,
            xref: 'x', yref: 'paper',
            line: { color: 'rgba(255,255,255,0.25)', width: 1, dash: 'solid' }
        });

        return {
            ...figure,
            layout: { ...figure.layout, shapes: shapes }
        };
    }
    """,
    Output("main-chart", "figure", allow_duplicate=True),
    Input("main-chart", "hoverData"),
    State("main-chart", "figure"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
