import os
import json
import time
from curl_cffi import requests as cffi_requests
import yfinance as yf
import requests
import pandas as pd
import numpy as np

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]
MOD = os.environ.get("TARAMA_MODU", "tarama")

session = cffi_requests.Session(impersonate="chrome")

POZISYON_DOSYASI = "pozisyonlar.json"
STOP_YUZDE = 0.03
HEDEF_YUZDE = 0.06
PARCA_BOYUTU = 40

def hisseleri_oku():
    with open("hisseler.txt", "r", encoding="utf-8") as f:
        return [s.strip() + ".IS" for s in f if s.strip()]

def pozisyonlari_oku():
    try:
        with open(POZISYON_DOSYASI, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def pozisyonlari_yaz(pozisyonlar):
    with open(POZISYON_DOSYASI, "w", encoding="utf-8") as f:
        json.dump(pozisyonlar, f, ensure_ascii=False, indent=2)

def telegram_gonder(mesaj):
    for i in range(0, len(mesaj), 3800):
        parca = mesaj[i:i+3800]
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": parca}
        )

def ema(seri, periyot):
    return seri.ewm(span=periyot, adjust=False).mean()

def rsi_hesapla(kapanis, periyot=14):
    delta = kapanis.diff()
    kazanc = delta.clip(lower=0)
    kayip = -delta.clip(upper=0)
    ort_kazanc = kazanc.ewm(alpha=1/periyot, adjust=False).mean()
    ort_kayip = kayip.ewm(alpha=1/periyot, adjust=False).mean()
    rs = ort_kazanc / ort_kayip
    return 100 - (100 / (1 + rs))

def macd_hesapla(kapanis):
    macd = ema(kapanis, 12) - ema(kapanis, 26)
    sinyal = ema(macd, 9)
    histogram = macd - sinyal
    return macd, sinyal, histogram

def adx_hesapla(yuksek, dusuk, kapanis, periyot=14):
    yukselis = yuksek.diff()
    dusus = -dusuk.diff()
    plus_dm = np.where((yukselis > dusus) & (yukselis > 0), yukselis, 0.0)
    minus_dm = np.where((dusus > yukselis) & (dusus > 0), dusus, 0.0)
    tr1 = yuksek - dusuk
    tr2 = (yuksek - kapanis.shift()).abs()
    tr3 = (dusuk - kapanis.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/periyot, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=yuksek.index).ewm(alpha=1/periyot, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=yuksek.index).ewm(alpha=1/periyot, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/periyot, adjust=False).mean()

def vwap_hesapla(df):
    gun = df.index.date
    tipik = (df['High'] + df['Low'] + df['Close']) / 3
    tv = tipik * df['Volume']
    cum_tv = pd.Series(tv.values, index=df.index).groupby(gun).cumsum()
    cum_vol = pd.Series(df['Volume'].values, index=df.index).groupby(gun).cumsum()
    return cum_tv / cum_vol

def obv_hesapla(kapanis, hacim):
    yon = np.sign(kapanis.diff().fillna(0))
    return (yon * hacim).cumsum()

def to_4h(df1s):
    df1s = df1s.copy()
    df1s['gun'] = df1s.index.date
    bloklar = []
    for gun, grup in df1s.groupby('gun'):
        grup = grup.sort_index()
        for i in range(0, len(grup), 4):
            blok = grup.iloc[i:i+4]
            if blok.empty:
                continue
            bloklar.append({
                'index': blok.index[-1],
                'Open': blok['Open'].iloc[0],
                'High': blok['High'].max(),
                'Low': blok['Low'].min(),
                'Close': blok['Close'].iloc[-1],
                'Volume': blok['Volume'].sum(),
            })
    if not bloklar:
        return pd.DataFrame()
    df4 = pd.DataFrame(bloklar).set_index('index')
    return df4

def kriterleri_degerlendir(df1s):
    df4 = to_4h(df1s)
    if len(df4) < 40:
        return None

    kapanis = df4['Close']
    ema8 = ema(kapanis, 8)
    ema21 = ema(kapanis, 21)
    if ema8.iloc[-1] <= ema21.iloc[-1]:
        return None

    macd, sinyal, hist = macd_hesapla(kapanis)
    rsi = rsi_hesapla(kapanis)
    adx = adx_hesapla(df4['High'], df4['Low'], kapanis)
    hacim_ort = df4['Volume'].rolling(20).mean()
    vwap = vwap_hesapla(df4)
    obv = obv_hesapla(kapanis, df4['Volume'])
    obv_ort = ema(obv, 20)

    kriterler = [
        macd.iloc[-1] > sinyal.iloc[-1],
        hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2],
        55 <= rsi.iloc[-1] <= 72,
        adx.iloc[-1] >= 20,
        df4['Volume'].iloc[-1] > hacim_ort.iloc[-1],
        kapanis.iloc[-1] > vwap.iloc[-1],
        obv.iloc[-1] > obv_ort.iloc[-1],
    ]
    if sum(kriterler) >= 5:
        guncel_fiyat = float(df1s['Close'].iloc[-1])
        return guncel_fiyat
    return None

def veri_indir(tickerlar):
    sonuc = {}
    for i in range(0, len(tickerlar), PARCA_BOYUTU):
        parca = tickerlar[i:i+PARCA_BOYUTU]
        try:
            veri = yf.download(
                tickers=" ".join(parca),
                period="60d", interval="1h",
                group_by="ticker", threads=True,
                progress=False, session=session,
                auto_adjust=False
            )
            for t in parca:
                try:
                    df = veri if len(parca) == 1 else veri[t]
                    df = df.dropna()
                    if not df.empty:
                        sonuc[t] = df
                except Exception:
                    pass
        except Exception as e:
            print(f"Parça hatası: {e}")
        time.sleep(1)
    return sonuc

def main():
    tickerlar = hisseleri_oku()
    pozisyonlar = pozisyonlari_oku()
    veriler = veri_indir(tickerlar)

    yeni_al_mesajlari = []
    kapanis_mesajlari = []

    for t, df in veriler.items():
        fiyat = float(df['Close'].iloc[-1])

        if t in pozisyonlar:
            poz = pozisyonlar[t]
            if fiyat <= poz['stop']:
                kapanis_mesajlari.append(
                    f"🔴 STOP: {t}\nGiriş: {poz['giris']:.2f} → Şu an: {fiyat:.2f} ({(fiyat/poz['giris']-1)*100:.1f}%)"
                )
                del pozisyonlar[t]
            elif fiyat >= poz['hedef']:
                kapanis_mesajlari.append(
                    f"🎯 HEDEF: {t}\nGiriş: {poz['giris']:.2f} → Şu an: {fiyat:.2f} ({(fiyat/poz['giris']-1)*100:.1f}%)"
                )
                del pozisyonlar[t]
            continue

        if MOD == "tarama":
            try:
                giris_fiyati = kriterleri_degerlendir(df)
            except Exception:
                giris_fiyati = None
            if giris_fiyati:
                hedef = giris_fiyati * (1 + HEDEF_YUZDE)
                stop = giris_fiyati * (1 - STOP_YUZDE)
                pozisyonlar[t] = {"giris": giris_fiyati, "hedef": hedef, "stop": stop}
                yeni_al_mesajlari.append(
                    f"🟢 AL: {t}\nGiriş: {giris_fiyati:.2f}\nHedef: {hedef:.2f} (+%{HEDEF_YUZDE*100:.0f})\nStop: {stop:.2f} (-%{STOP_YUZDE*100:.0f})"
                )

    mesaj = ""
    if yeni_al_mesajlari:
        mesaj += "📈 YENİ SİNYALLER\n\n" + "\n\n".join(yeni_al_mesajlari) + "\n\n"
    if kapanis_mesajlari:
        mesaj += "📌 POZİSYON GÜNCELLEMESİ\n\n" + "\n\n".join(kapanis_mesajlari) + "\n\n"

    if MOD == "ozet":
        if pozisyonlar:
            satirlar = []
            for t, poz in pozisyonlar.items():
                fiyat = float(veriler[t]['Close'].iloc[-1]) if t in veriler else poz['giris']
                degisim = (fiyat / poz['giris'] - 1) * 100
                satirlar.append(f"{t}: Giriş {poz['giris']:.2f} → Şu an {fiyat:.2f} ({degisim:+.1f}%)")
            mesaj += "🗓️ GÜN SONU AÇIK POZİSYONLAR\n\n" + "\n".join(satirlar)
        else:
            mesaj += "🗓️ Gün sonu: açık pozisyon yok."

    if not mesaj:
        mesaj = "Bu taramada yeni sinyal veya pozisyon güncellemesi yok."

    telegram_gonder(mesaj)
    pozisyonlari_yaz(pozisyonlar)

if __name__ == "__main__":
    main()
