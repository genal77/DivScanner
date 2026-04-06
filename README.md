# DivScanner

A real-time BTC divergence scanner that aggregates Cumulative Volume Delta (CVD) across multiple spot and futures exchanges, detects price/CVD divergences, and delivers alerts to Telegram.

## What it does

DivScanner continuously collects order flow data (OHLC + taker buy volume) from 10+ exchanges, computes aggregated CVD, and scans for divergences between price action and buying/selling pressure. When a divergence is detected, it sends a Telegram alert with a chart screenshot.

**Detected signal types:**
| Signal | Meaning |
|---|---|
| 🔵 Selling Exhaustion | Price makes lower low, CVD makes higher low — sellers losing momentum |
| 🟢 Selling Absorption | Price makes higher low, CVD makes lower low — buyers absorbing sell pressure |
| 🟠 Buying Exhaustion | Price makes higher high, CVD makes lower high — buyers losing momentum |
| 🔴 Buying Absorption | Price makes lower high, CVD makes higher high — sellers absorbing buy pressure |

## Architecture

```
collector.py  →  data/*.parquet  →  app.py (dashboard)
                                →  analysis.py (divergence detection)
                                →  Telegram alerts (screenshot + text)
```

- **collector.py** — fetches klines and trades every 60s from all exchanges, writes Parquet files
- **analysis.py** — shared logic: CVD computation, pivot detection, divergence detection, figure building
- **app.py** — Dash dashboard, refreshes every 10s
- **outcome_tracker.py** — tracks signal outcomes for backtesting

## Dashboard

Five panels, shared x-axis:
1. BTC/USDT price (Binance Spot)
2. Delta per candle (aggregated spot)
3. CVD Spot aggregated (selectable exchanges)
4. CVD Futures aggregated (selectable exchanges)
5. Open Interest (Binance Futures)

Active divergences are highlighted with lines on the price and CVD panels, and summarised in a banner at the top.

## Telegram alerts

Alerts fire when a live divergence is detected on any of the monitored timeframes (5m, 15m, 30m, 1h). Each alert includes:
- Signal type, timeframes, and direction
- Price move in ATR units, CVD move in sigma units
- Open Interest change between pivots
- Persistence score (% of candles in the window where price and CVD diverged)
- 540×960 chart screenshot (mobile-optimised, 9:16)

## Exchanges

**Spot CVD:**
Binance, MEXC, Bybit, OKX, Gate.io, Coinbase, Kraken

**Futures CVD:**
Binance Futures, Bybit Futures, OKX Futures, Gate.io Futures, MEXC Futures

**Open Interest:**
Binance Futures

## Stack

- Python 3.9
- Dash + Plotly (dashboard)
- Pandas + PyArrow (data storage)
- Kaleido + Chromium (chart rendering for Telegram screenshots)
- Docker + Docker Compose (deployment)

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ammarcinkowski-max/DivScanner.git
cd DivScanner
```

### 2. Create `.env`

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
DASH_USERNAME=admin
DASH_PASSWORD=your_password_here
```

To create a Telegram bot: talk to [@BotFather](https://t.me/BotFather) on Telegram.

### 3. Run with Docker

```bash
docker compose up -d --build
```

Dashboard will be available at `http://localhost:8050`.

### 4. Run locally (without Docker)

```bash
pip install -r requirements.txt
python3 collector.py &
python3 app.py
```

## Deployment (VPS)

```bash
# On the VPS
git clone https://github.com/ammarcinkowski-max/DivScanner.git
cd DivScanner
cp .env.example .env
# edit .env with your values
docker compose up -d --build
```

After code updates:

```bash
git pull
docker compose restart collector app   # for .py changes
docker compose up -d --build           # only if Dockerfile or requirements.txt changed
```

## Data

All data is stored locally in `data/` as Parquet files (excluded from the repo via `.gitignore`):

| File | Contents |
|---|---|
| `btc_spot_5m.parquet` | Spot klines per exchange |
| `btc_futures_5m.parquet` | Futures klines per exchange |
| `btc_oi_1m.parquet` | Open Interest snapshots |
| `signal_log.csv` | Historical signal log for backtesting |
| `alert_state.json` | Persists last-alerted candle timestamps |
