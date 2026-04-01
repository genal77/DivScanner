"""
check_units.py — Sprawdza jednostki wolumenu dla każdej giełdy (spot + futures).

Dla każdej giełdy pobiera 3 ostatnie transakcje i wyświetla:
- surowe pola z API
- nasze obliczone 'size' i 'is_buyer'
- szacunkową wartość w BTC (żeby wykryć błędy jednostek)

Uruchom: python3 check_units.py
"""

import requests

APPROX_BTC_PRICE = 83_000  # przybliżona cena BTC do oceny czy wartości mają sens


def header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def show_trades(trades: list, size_col: str, side_col: str, time_col: str, note: str = ""):
    """Wyświetla pierwsze 3 transakcje z podanymi kolumnami."""
    if note:
        print(f"  UWAGA: {note}")
    for t in trades[:3]:
        size = float(t.get(size_col, 0))
        side = t.get(side_col, "?")
        ts   = t.get(time_col, "?")
        btc_equiv = size if size < 10_000 else f"⚠️  PODEJRZANE — {size:.0f} (prawdopodobnie nie BTC)"
        print(f"  size={size:>12.4f}  side={side:<5}  ts={ts}  → BTC equiv: {btc_equiv}")


# ---------------------------------------------------------------------------
# SPOT
# ---------------------------------------------------------------------------

header("BINANCE SPOT — klines taker_buy_vol (5m, ostatnia świeca)")
try:
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": "BTCUSDT", "interval": "5m", "limit": 1}, timeout=10).json()
    candle = r[0]
    vol          = float(candle[5])
    taker_buy    = float(candle[9])
    taker_sell   = vol - taker_buy
    delta        = taker_buy - taker_sell
    print(f"  volume={vol:.2f} BTC  taker_buy={taker_buy:.2f} BTC  taker_sell={taker_sell:.2f} BTC  delta={delta:.2f}")
    print(f"  USD value ≈ ${vol * APPROX_BTC_PRICE:,.0f}  {'✅ OK' if vol < 5000 else '⚠️  sprawdź'}")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("MEXC SPOT — trades (qty, isBuyerMaker)")
try:
    r = requests.get("https://api.mexc.com/api/v3/trades",
                     params={"symbol": "BTCUSDT", "limit": 3}, timeout=10).json()
    print(f"  Surowe pola: {list(r[0].keys())}")
    for t in r[:3]:
        size = float(t["qty"])
        side = "buy" if not t["isBuyerMaker"] else "sell"
        print(f"  qty={size:.6f}  isBuyerMaker={t['isBuyerMaker']}  → taker={side}  BTC≈{size:.4f}")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("BYBIT SPOT — trades (size, side)")
try:
    r = requests.get("https://api.bybit.com/v5/market/recent-trade",
                     params={"category": "spot", "symbol": "BTCUSDT", "limit": 3}, timeout=10).json()
    trades = r["result"]["list"]
    print(f"  Surowe pola: {list(trades[0].keys())}")
    show_trades(trades, "size", "side", "time")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("OKX SPOT — trades (sz, side)")
try:
    r = requests.get("https://www.okx.com/api/v5/market/trades",
                     params={"instId": "BTC-USDT", "limit": 3}, timeout=10).json()
    trades = r["data"]
    print(f"  Surowe pola: {list(trades[0].keys())}")
    show_trades(trades, "sz", "side", "ts")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("GATE SPOT — trades (amount, side)")
try:
    r = requests.get("https://api.gateio.ws/api/v4/spot/trades",
                     params={"currency_pair": "BTC_USDT", "limit": 3}, timeout=10).json()
    print(f"  Surowe pola: {list(r[0].keys())}")
    show_trades(r, "amount", "side", "create_time")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("COINBASE — trades (size, side)")
try:
    r = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/trades",
                     params={"limit": 3}, timeout=10).json()
    print(f"  Surowe pola: {list(r[0].keys())}")
    show_trades(r, "size", "side", "time")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("KRAKEN — trades (volume, side)")
try:
    r = requests.get("https://api.kraken.com/0/public/Trades",
                     params={"pair": "XBTUSD"}, timeout=10).json()
    trades = r["result"]["XXBTZUSD"]
    print(f"  Format: [price, volume, time, side, order_type, misc, trade_id]")
    for t in trades[:3]:
        size = float(t[1])
        side = "buy" if t[3] == "b" else "sell"
        print(f"  volume={size:.6f}  side={side}  BTC≈{size:.4f}")
except Exception as e:
    print(f"  BŁĄD: {e}")


# ---------------------------------------------------------------------------
# FUTURES
# ---------------------------------------------------------------------------

header("BINANCE FUTURES — klines taker_buy_vol (5m, ostatnia świeca)")
try:
    r = requests.get("https://fapi.binance.com/fapi/v1/klines",
                     params={"symbol": "BTCUSDT", "interval": "5m", "limit": 1}, timeout=10).json()
    candle = r[0]
    vol       = float(candle[5])
    taker_buy = float(candle[9])
    delta     = taker_buy - (vol - taker_buy)
    print(f"  volume={vol:.2f} BTC  taker_buy={taker_buy:.2f} BTC  delta={delta:.2f}")
    print(f"  USD value ≈ ${vol * APPROX_BTC_PRICE:,.0f}  {'✅ OK' if vol < 50_000 else '⚠️  sprawdź'}")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("BYBIT FUTURES (linear) — trades (size, side)")
try:
    r = requests.get("https://api.bybit.com/v5/market/recent-trade",
                     params={"category": "linear", "symbol": "BTCUSDT", "limit": 3}, timeout=10).json()
    trades = r["result"]["list"]
    print(f"  Surowe pola: {list(trades[0].keys())}")
    show_trades(trades, "size", "side", "time",
                note="Bybit linear: size w BTC czy kontraktach? Sprawdź czy ~0.001-10 BTC")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("OKX FUTURES (BTC-USDT-SWAP) — trades (sz, side)")
try:
    r = requests.get("https://www.okx.com/api/v5/market/trades",
                     params={"instId": "BTC-USDT-SWAP", "limit": 3}, timeout=10).json()
    trades = r["data"]
    print(f"  Surowe pola: {list(trades[0].keys())}")
    show_trades(trades, "sz", "side", "ts",
                note="OKX SWAP: 1 kontrakt = 0.01 BTC. Jeśli sz=500 → 5 BTC")
    print(f"  Przeliczenie: sz × 0.01 = BTC")
    for t in trades[:3]:
        sz = float(t["sz"])
        print(f"    sz={sz:.0f}  →  {sz * 0.01:.4f} BTC  (≈ ${sz * 0.01 * APPROX_BTC_PRICE:,.0f})")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("GATE FUTURES (BTC_USDT) — trades (size, price)")
try:
    r = requests.get("https://api.gateio.ws/api/v4/futures/usdt/trades",
                     params={"contract": "BTC_USDT", "limit": 3}, timeout=10).json()
    print(f"  Surowe pola: {list(r[0].keys())}")
    note = "Gate futures: 1 kontrakt = 1 USD. size/price = BTC"
    print(f"  UWAGA: {note}")
    for t in r[:3]:
        size  = abs(float(t["size"]))
        price = float(t["price"])
        btc   = size / price
        side  = "buy" if float(t["size"]) > 0 else "sell"
        print(f"    size={size:.0f}  price={price:.0f}  side={side}  →  {btc:.6f} BTC  (≈ ${size:.0f})")
except Exception as e:
    print(f"  BŁĄD: {e}")


header("MEXC FUTURES (BTC_USDT) — trades (v, T)")
try:
    r = requests.get("https://contract.mexc.com/api/v1/contract/deals/BTC_USDT",
                     params={"limit": 3}, timeout=10).json()
    trades = r["data"]
    print(f"  Surowe pola: {list(trades[0].keys())}")
    note = "MEXC futures: sprawdź czy 'v' to BTC, USD czy kontrakty"
    print(f"  UWAGA: {note}")
    for t in trades[:3]:
        v    = float(t["v"])
        p    = float(t["p"])
        side = "buy" if t["T"] == 1 else "sell"
        print(f"    v={v:.4f}  p={p:.0f}  T={t['T']}({side})  → jeśli BTC: {v:.4f} BTC  jeśli USD: {v/p:.6f} BTC")
except Exception as e:
    print(f"  BŁĄD: {e}")

print(f"\n{'='*60}")
print("  KONIEC — porównaj wartości z oczekiwanymi rzędami wielkości:")
print("  Typowy trade BTC: 0.001 – 5 BTC (~$80 – $400,000)")
print(f"{'='*60}\n")
