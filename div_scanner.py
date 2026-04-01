import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

def start():
    print("--- SKANER AKTYWNY: Dopasowany obszar wykresu | 200 świec ---")
    
    while True:
        try:
            # 1. POBIERANIE DANYCH (200 świeczek 15m)
            url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=200"
            data = requests.get(url).json()
            df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Vol', 'C_T', 'QAV', 'N_T', 'Taker_B', 'Taker_Q', 'Ign'])
            cols = ['Open', 'High', 'Low', 'Close', 'Vol', 'Taker_B']
            df[cols] = df[cols].apply(pd.to_numeric)
            
            # Czas warszawski (Letni +2h)
            df['Time'] = pd.to_datetime(df['Time'], unit='ms') + timedelta(hours=2) 
            now_warsaw = datetime.now().strftime('%H:%M:%S')

            # 2. OBLICZENIA CVD
            df['Delta'] = df['Taker_B'] - (df['Vol'] - df['Taker_B'])
            df['CVD_Close'] = df['Delta'].cumsum()
            df['CVD_Open'] = df['CVD_Close'].shift(1).fillna(df['CVD_Close'].iloc[0])
            df['CVD_High'] = df[['CVD_Open', 'CVD_Close']].max(axis=1)
            df['CVD_Low'] = df[['CVD_Open', 'CVD_Close']].min(axis=1)

            # 3. SZUKANIE PIVOTÓW
            win = 5
            df['P_Low'] = False
            df['P_High'] = False
            for i in range(win, len(df) - win):
                if df['Low'].iloc[i] == df['Low'].iloc[i-win : i+win+1].min(): df.at[df.index[i], 'P_Low'] = True
                if df['High'].iloc[i] == df['High'].iloc[i-win : i+win+1].max(): df.at[df.index[i], 'P_High'] = True

            # 4. TWORZENIE WYKRESU
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.4])
            
            fig.add_trace(go.Candlestick(x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Cena"), row=1, col=1)
            fig.add_trace(go.Candlestick(x=df['Time'], open=df['CVD_Open'], high=df['CVD_High'], low=df['CVD_Low'], close=df['CVD_Close'], name="CVD"), row=2, col=1)

            # SEPARATORY DNI
            days = df['Time'].dt.date.unique()
            for day in days:
                day_start = datetime.combine(day, datetime.min.time())
                if day_start > df['Time'].iloc[0]: # Rysuj tylko jeśli mieści się w widocznym zakresie
                    fig.add_vline(x=day_start, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.2)")

            p_lows = df.index[df['P_Low']].tolist()
            p_highs = df.index[df['P_High']].tolist()
            curr = len(df) - 1
            active_signal = ""
            signal_color = "white"

            # RYSOWANIE DYWERGENCJI (HISTORYCZNE)
            for i in range(1, len(p_lows)):
                p1, p2 = p_lows[i-1], p_lows[i]
                pdif, cdif = df['Low'].iloc[p2] - df['Low'].iloc[p1], df['CVD_Low'].iloc[p2] - df['CVD_Low'].iloc[p1]
                if pdif < 0 and cdif > 0: color, dash = "cyan", "dash"
                elif pdif > 0 and cdif < 0: color, dash = "cyan", "solid"
                else: continue
                for r in [1, 2]:
                    y0, y1 = df['Low' if r==1 else 'CVD_Low'].iloc[p1], df['Low' if r==1 else 'CVD_Low'].iloc[p2]
                    fig.add_shape(type="line", x0=df['Time'][p1], y0=y0, x1=df['Time'][p2], y1=y1, line=dict(color=color, width=2, dash=dash), row=r, col=1)

            for i in range(1, len(p_highs)):
                p1, p2 = p_highs[i-1], p_highs[i]
                pdif, cdif = df['High'].iloc[p2] - df['High'].iloc[p1], df['CVD_High'].iloc[p2] - df['CVD_High'].iloc[p1]
                if pdif > 0 and cdif < 0: color, dash = "magenta", "dash"
                elif pdif < 0 and cdif > 0: color, dash = "magenta", "solid"
                else: continue
                for r in [1, 2]:
                    y0, y1 = df['High' if r==1 else 'CVD_High'].iloc[p1], df['High' if r==1 else 'CVD_High'].iloc[p2]
                    fig.add_shape(type="line", x0=df['Time'][p1], y0=y0, x1=df['Time'][p2], y1=y1, line=dict(color=color, width=2, dash=dash), row=r, col=1)

            # LOGIKA LIVE
            if p_lows and df['Low'].iloc[p_lows[-1]+1 : curr+1].min() == df['Low'].iloc[curr]:
                last_l = p_lows[-1]
                l_pdif, l_cdif = df['Low'].iloc[curr] - df['Low'].iloc[last_l], df['CVD_Low'].iloc[curr] - df['CVD_Low'].iloc[last_l]
                if l_pdif < 0 and l_cdif > 0: active_signal, signal_color = "SELLERS EXHAUSTION", "cyan"
                elif l_pdif > 0 and l_cdif < 0: active_signal, signal_color = "SELLERS ABSORPTION (PASSIVE BUYING)", "cyan"
                if active_signal:
                    for r in [1, 2]:
                        y0, y1 = df['Low' if r==1 else 'CVD_Low'].iloc[last_l], df['Low' if r==1 else 'CVD_Low'].iloc[curr]
                        fig.add_shape(type="line", x0=df['Time'][last_l], y0=y0, x1=df['Time'][curr], y1=y1, line=dict(color="cyan", width=4, dash="dash" if "EXHAUSTION" in active_signal else "solid"), row=r, col=1)

            if p_highs and df['High'].iloc[p_highs[-1]+1 : curr+1].max() == df['High'].iloc[curr]:
                last_h = p_highs[-1]
                h_pdif, h_cdif = df['High'].iloc[curr] - df['High'].iloc[last_h], df['CVD_High'].iloc[curr] - df['CVD_High'].iloc[last_h]
                if h_pdif > 0 and h_cdif < 0: active_signal, signal_color = "BUYERS EXHAUSTION", "magenta"
                elif h_pdif < 0 and h_cdif > 0: active_signal, signal_color = "BUYERS ABSORPTION (PASSIVE SELLING)", "magenta"
                if active_signal:
                    for r in [1, 2]:
                        y0, y1 = df['High' if r==1 else 'CVD_High'].iloc[last_h], df['High' if r==1 else 'CVD_High'].iloc[curr]
                        fig.add_shape(type="line", x0=df['Time'][last_h], y0=y0, x1=df['Time'][curr], y1=y1, line=dict(color="magenta", width=4, dash="dash" if "EXHAUSTION" in active_signal else "solid"), row=r, col=1)

            # 5. ANNOTACJE I ALERT PANEL
            fig.add_annotation(x=0.01, y=0.99, xref="paper", yref="paper", text=f"<b>BTC/USDT Binance | {now_warsaw}</b>", showarrow=False, font=dict(size=13, color="white"), align="left")
            fig.add_annotation(x=0.01, y=0.38, xref="paper", yref="paper", text="<b>CVD Spot Binance</b>", showarrow=False, font=dict(size=13, color="white"), align="left")
            
            if active_signal:
                fig.add_annotation(x=0.5, y=0.98, xref="paper", yref="paper", text=f"<b>{active_signal}</b>", showarrow=False, font=dict(size=20, color=signal_color), bgcolor="rgba(0,0,0,0.85)", bordercolor=signal_color, borderpad=10)
                print(f"!!! ALERT: {active_signal} na {df['Close'].iloc[curr]}")

            # 6. DOPASOWANIE OSI (USUNIĘCIE PUSTEGO MIEJSCA)
            # Ustawiamy zakres od pierwszej do ostatniej świecy + mały margines na prawo
            x_start = df['Time'].iloc[0]
            x_end = df['Time'].iloc[-1] + timedelta(minutes=60) # 1 godzina zapasu na prawo

            fig.update_xaxes(
                range=[x_start, x_end], 
                rangeslider_visible=False, 
                gridcolor="#222", 
                tickformat="%H:%M\n%d/%m", 
                dtick=7200000
            )
            fig.update_yaxes(side="right", gridcolor="#222")
            fig.update_layout(template="plotly_dark", autosize=True, margin=dict(l=10, r=60, t=30, b=10), showlegend=False)

            html_content = fig.to_html(config={'responsive': True}, include_plotlyjs='cdn', full_html=False)
            with open("scanner_wykres3.html", "w") as f:
                f.write(f"<html><head><meta http-equiv='refresh' content='30'><style>body, html {{ margin: 0; padding: 0; height: 100%; overflow: hidden; background-color: #111; }} #chart {{ height: 100vh; width: 100vw; }}</style></head><body><div id='chart'>{html_content}</div></body></html>")

            time.sleep(30)
            
        except Exception as e:
            print(f"Błąd: {e}")
            time.sleep(10)

if __name__ == "__main__":
    start()