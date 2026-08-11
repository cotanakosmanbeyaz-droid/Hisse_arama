import os
import json
import time
from datetime import datetime
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
GUNLUK_DOSYASI = "islem_gecmisi.csv"
PARCA_BOYUTU = 30

PIVOT_PENCERE = 5
DIP_TOLERANS = 0.03
MIN_DIP_ARALIGI = 8
MIN_TEPE_YUKSEKLIK = 0.03
LOOKBACK_BAR = 360


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


def islem_gunlugune_yaz(sembol, karar, giris_zaman, giris_fiyat, cikis_zaman, cikis_fiyat, getiri, nedenler):
    dosya_var = os.path.exists(GUNLUK_DOSYASI)
    with open(GUNLUK_DOSYASI, "a", encoding="utf-8", newline="") as f:
        if not dosya_var:
            f.write("sembol,karar,giris_zaman,giris_fiyat,cikis_zaman,cikis_fiyat,getiri_%,nedenler\n")
        satir = [sembol, karar, giris_zaman, str(giris_fiyat), cikis_zaman, str(cikis_fiyat), str(getiri), "; ".join(nedenler)]
        satir = [f'"{a}"' if "," in a else a for a in satir]
        f.write(",".join(satir) + "\n")


def telegram_gonder(mesaj):
    for i in range(0, len(mesaj), 3800):
        parca = mesaj[i:i+3800]
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": parca, "parse_mode": "HTML"}
        )


def ema_hesapla(seri, periyot):
    return seri.ewm(span=periyot, adjust=False).mean()


def sma_hesapla(seri, periyot):
    return seri.rolling(periyot).mean()


def rsi_hesapla(kapanis, periyot=14):
    delta = kapanis.diff()
    kazanc = delta.clip(lower=0)
    kayip = -delta.clip(upper=0)
    ort_kazanc = kazanc.ewm(alpha=1/periyot, min_periods=periyot).mean()
    ort_kayip = kayip.ewm(alpha=1/periyot, min_periods=periyot).mean()
    rs = ort_kazanc / ort_kayip
    return 100 - (100 / (1 + rs))


def macd_hesapla(kapanis, hizli=12, yavas=26, sinyal=9):
    ema_hizli = kapanis.ewm(span=hizli, adjust=False).mean()
    ema_yavas = kapanis.ewm(span=yavas, adjust=False).mean()
    macd_line = ema_hizli - ema_yavas
    sinyal_line = macd_line.ewm(span=sinyal, adjust=False).mean()
    return macd_line, sinyal_line, macd_line - sinyal_line


def atr_hesapla(df, periyot=14):
    onceki = df["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-onceki).abs(), (df["Low"]-onceki).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/periyot, min_periods=periyot).mean()


def adx_hesapla(df, periyot=14):
    yuksek, dusuk, kapanis = df["High"], df["Low"], df["Close"]
    onceki = kapanis.shift(1)
    tr = pd.concat([yuksek-dusuk, (yuksek-onceki).abs(), (dusuk-onceki).abs()], axis=1).max(axis=1)
    yuk_har = yuksek.diff()
    dus_har = -dusuk.diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(yuk_har > dus_har) & (yuk_har > 0)] = yuk_har
    minus_dm[(dus_har > yuk_har) & (dus_har > 0)] = dus_har
    atr = tr.ewm(alpha=1/periyot, min_periods=periyot).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/periyot, min_periods=periyot).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/periyot, min_periods=periyot).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1/periyot, min_periods=periyot).mean()


def bollinger_orta(kapanis, periyot=20):
    return kapanis.rolling(periyot).mean()


def fibonacci_seviyeleri(df, lookback=50):
    tepe = df["High"].rolling(lookback).max()
    dip = df["Low"].rolling(lookback).min()
    fark = tepe - dip
    return tepe, tepe - fark*0.382, tepe - fark*0.618


def pivot_dipleri_bul(df, pencere=PIVOT_PENCERE):
    low = df["Low"].to_numpy()
    n = len(df)
    sonuc = pd.Series(False, index=df.index)
    for i in range(pencere, n - pencere):
        dilim = low[i-pencere:i+pencere+1]
        if low[i] == dilim.min() and (dilim == low[i]).sum() == 1:
            sonuc.iloc[i] = True
    return sonuc


def uclu_dip_kontrol_et(df):
    if len(df) < LOOKBACK_BAR:
        return None
    pencere_df = df.tail(LOOKBACK_BAR).copy()
    pivotlar = pivot_dipleri_bul(pencere_df)
    dip_idx = list(pencere_df.index[pivotlar])
    if len(dip_idx) < 3:
        return None
    son3 = dip_idx[-3:]
    fiyatlar = [pencere_df.loc[z, "Low"] for z in son3]
    ort = sum(fiyatlar) / 3
    if any(abs(f-ort)/ort > DIP_TOLERANS for f in fiyatlar):
        return None
    p1, p2, p3 = [pencere_df.index.get_loc(z) for z in son3]
    if (p2-p1) < MIN_DIP_ARALIGI or (p3-p2) < MIN_DIP_ARALIGI:
        return None
    tepe1 = pencere_df["High"].iloc[p1:p2+1].max()
    tepe2 = pencere_df["High"].iloc[p2:p3+1].max()
    if tepe1 < ort*(1+MIN_TEPE_YUKSEKLIK) or tepe2 < ort*(1+MIN_TEPE_YUKSEKLIK):
        return None
    boyun = min(tepe1, tepe2)
    son_pencere = pencere_df.iloc[p3:]
    if len(son_pencere) < 2 or len(son_pencere) > 15:
        return None
    kirilim_oncesi = son_pencere["Close"].iloc[:-1] <= boyun
    kirilim_simdi = son_pencere["Close"].iloc[-1] > boyun
    if not (kirilim_oncesi.all() and kirilim_simdi):
        return None
    return round(float(boyun), 4)


def puanlari_hesapla(df):
    df = df.copy()
    df["EMA8"] = ema_hesapla(df["Close"], 8)
    df["EMA21"] = ema_hesapla(df["Close"], 21)
    df["EMA50"] = ema_hesapla(df["Close"], 50)
    df["SMA200"] = sma_hesapla(df["Close"], 200)
    df["RSI"] = rsi_hesapla(df["Close"])
    df["MACD"], df["MACD_Sinyal"], df["MACD_Hist"] = macd_hesapla(df["Close"])
    df["ATR"] = atr_hesapla(df)
    df["ADX"] = adx_hesapla(df)
    df["OrtaBant"] = bollinger_orta(df["Close"])
    df["HacimOrt20"] = sma_hesapla(df["Volume"], 20)
    df["ATROrt5"] = df["ATR"].rolling(5).mean()
    df["FibTepe"], df["Fib382"], df["Fib618"] = fibonacci_seviyeleri(df)

    puan = pd.DataFrame(index=df.index)
    puan["ema_dizilim"] = ((df["EMA8"] > df["EMA21"]) & (df["EMA21"] > df["EMA50"])).astype(int) * 20
    kesisim = (df["EMA8"].shift(1) < df["EMA21"].shift(1)) & (df["EMA8"] > df["EMA21"])
    puan["ema_kesisim"] = kesisim.rolling(5, min_periods=1).max().fillna(0).astype(int) * 10
    puan["macd_uzerinde"] = (df["MACD"] > df["MACD_Sinyal"]).astype(int) * 15
    puan["macd_hist"] = (df["MACD_Hist"] > 0).astype(int) * 5
    puan["rsi"] = ((df["RSI"] >= 55) & (df["RSI"] <= 70)).astype(int) * 10
    puan["hacim"] = (df["Volume"] > df["HacimOrt20"]).astype(int) * 15
    puan["adx"] = (df["ADX"] > 25).astype(int) * 10
    puan["bollinger"] = (df["Close"] > df["OrtaBant"]).astype(int) * 5
    tolerans = df["FibTepe"] * 0.01
    fib382 = (abs(df["Low"]-df["Fib382"]) <= tolerans) & (df["Close"] > df["Fib382"])
    fib618 = (abs(df["Low"]-df["Fib618"]) <= tolerans) & (df["Close"] > df["Fib618"])
    puan["fibonacci"] = (fib382 | fib618).rolling(3, min_periods=1).max().fillna(0).astype(int) * 5
    puan["atr"] = (df["ATR"] > df["ATROrt5"]).astype(int) * 5

    df["Skor"] = puan.sum(axis=1)

    def karar_belirle(s):
        if pd.isna(s):
            return None
        if s >= 85:
            return "GÜÇLÜ AL"
        if s >= 80:
            return "AL"
        if s >= 55:
            return "İZLEME LİSTESİ"
        return "İŞLEM YOK"

    df["Karar"] = df["Skor"].apply(karar_belirle)

    df["Cikis_EMA"] = df["EMA8"] < df["EMA21"]
    df["Cikis_MACD"] = (df["MACD"].shift(1) > df["MACD_Sinyal"].shift(1)) & (df["MACD"] < df["MACD_Sinyal"])
    df["Cikis_RSI"] = df["RSI"] < 50
    df["Cikis_Hacim"] = (df["Close"] < df["Open"]) & (df["Volume"] > df["HacimOrt20"])
    df["Cikis_EMA50Alti"] = df["Close"] < df["EMA50"]
    return df


def toplu_indir(tickerlar, period, interval):
    sonuc = {}
    for i in range(0, len(tickerlar), PARCA_BOYUTU):
        parca = tickerlar[i:i+PARCA_BOYUTU]
        try:
            veri = yf.download(
                tickers=" ".join(parca), period=period, interval=interval,
                group_by="ticker", threads=True, progress=False,
                session=session, auto_adjust=False
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
            print(f"Parça hatası ({interval}): {e}")
        time.sleep(1)
    return sonuc


def guncel_fiyat_al(tickerlar):
    if not tickerlar:
        return {}
    veriler = toplu_indir(tickerlar, period="1d", interval="1m")
    return {t: float(df['Close'].iloc[-1]) for t, df in veriler.items()}


def main():
    tickerlar = hisseleri_oku()
    pozisyonlar = pozisyonlari_oku()
    ham_veri = toplu_indir(tickerlar, period="729d", interval="1h")

    skorlu = {}
    for t, df1s in ham_veri.items():
        try:
            df4 = df1s.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna()
            if len(df4) < 210:
                continue
            df4 = puanlari_hesapla(df4)
            df4 = df4.dropna(subset=["Skor", "ATR", "SMA200"])
            if df4.empty:
                continue
            skorlu[t] = df4
        except Exception as e:
            print(f"Skor hatası {t}: {e}")

    zaman_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    if MOD == "tarama":
        adaylar = []
        for t, df4 in skorlu.items():
            if t in pozisyonlar:
                continue
            son = df4.iloc[-1]
            if son["Karar"] in ("GÜÇLÜ AL", "AL"):
                adaylar.append(t)

        acik_liste = [t for t in pozisyonlar.keys() if t in skorlu]
        guncel_gerekli = list(set(adaylar) | set(acik_liste))
        guncel_fiyatlar = guncel_fiyat_al(guncel_gerekli)

        giris_mesajlari = []
        cikis_mesajlari = []
        uclu_dip_mesajlari = []

        for t in acik_liste:
            df4 = skorlu[t]
            son = df4.iloc[-1]
            poz = pozisyonlar[t]
            fiyat = guncel_fiyatlar.get(t, float(son["Close"]))

            nedenler = []
            if bool(son["Cikis_EMA"]):
                nedenler.append("EMA8, EMA21'in altına indi")
            if bool(son["Cikis_MACD"]):
                nedenler.append("MACD aşağı kesişim yaptı")
            if bool(son["Cikis_RSI"]):
                nedenler.append("RSI 50'nin altına düştü")
            if bool(son["Cikis_Hacim"]):
                nedenler.append("Hacim satış yönünde belirgin arttı")
            if bool(son["Cikis_EMA50Alti"]):
                nedenler.append("Fiyat EMA50'nin altında kapandı")
            if fiyat <= poz["stop_loss"]:
                nedenler.append(f"Stop-loss tetiklendi ({poz['stop_loss']:.2f})")
            if fiyat >= poz["hedef"]:
                nedenler.append(f"Kar hedefine ulaşıldı ({poz['hedef']:.2f})")

            if nedenler:
                getiri = (fiyat - poz["giris_fiyat"]) / poz["giris_fiyat"] * 100
                cikis_mesajlari.append(
                    f"‼️ <b>{t}</b> - ÇIKIŞ ÖNERİSİ\n"
                    f"Fiyat: {fiyat:.2f} | Giriş: {poz['giris_fiyat']:.2f} | Getiri: %{getiri:.2f}\n"
                    f"Nedenler:\n" + "\n".join(f"- {n}" for n in nedenler)
                )
                islem_gunlugune_yaz(t, poz.get("karar", "?"), poz.get("zaman", ""), poz["giris_fiyat"],
                                     zaman_str, fiyat, round(getiri, 2), nedenler)
                del pozisyonlar[t]

        for t in adaylar:
            df4 = skorlu[t]
            son = df4.iloc[-1]
            fiyat = guncel_fiyatlar.get(t, float(son["Close"]))
            atr = float(son["ATR"])
            stop = fiyat - 2 * atr
            hedef = fiyat + 2 * (fiyat - stop)
            skor = int(son["Skor"])
            karar = son["Karar"]
            pozisyonlar[t] = {
                "giris_fiyat": fiyat, "stop_loss": round(stop, 4), "hedef": round(hedef, 4),
                "zaman": zaman_str, "karar": karar,
            }
            emoji = "🔴" if karar == "GÜÇLÜ AL" else "⚫"
            giris_mesajlari.append(
                f"{emoji} <b>{t}</b> - {karar}\n"
                f"Skor: {skor}/100\n"
                f"Giriş: {fiyat:.2f}\nStop: {stop:.2f}\nHedef: {hedef:.2f}"
            )

        for t, df4 in skorlu.items():
            try:
                boyun = uclu_dip_kontrol_et(df4)
                if boyun:
                    fiyat = guncel_fiyatlar.get(t, float(df4['Close'].iloc[-1]))
                    uclu_dip_mesajlari.append(
                        f"🔵 <b>{t}</b> - ÜÇLÜ DİP KIRILIMI\nFiyat: {fiyat:.2f}\nBoyun çizgisi: {boyun}"
                    )
            except Exception:
                pass

        mesaj = ""
        if giris_mesajlari:
            mesaj += "📈 YENİ SİNYALLER\n\n" + "\n\n".join(giris_mesajlari) + "\n\n"
        if cikis_mesajlari:
            mesaj += "📌 ÇIKIŞ ÖNERİLERİ\n\n" + "\n\n".join(cikis_mesajlari) + "\n\n"
        if uclu_dip_mesajlari:
            mesaj += "🔵 ÜÇLÜ DİP SİNYALLERİ\n\n" + "\n\n".join(uclu_dip_mesajlari)
        if not mesaj:
            mesaj = "Bu taramada yeni sinyal yok."

        telegram_gonder(mesaj)
        pozisyonlari_yaz(pozisyonlar)

    elif MOD == "ozet":
        if pozisyonlar:
            ozet_gerekli = list(pozisyonlar.keys())
            guncel_fiyatlar = guncel_fiyat_al(ozet_gerekli)
            satirlar = []
            for t, poz in pozisyonlar.items():
                fiyat = guncel_fiyatlar.get(t, poz["giris_fiyat"])
                degisim = (fiyat / poz["giris_fiyat"] - 1) * 100
                satirlar.append(f"{t}: Giriş {poz['giris_fiyat']:.2f} → Şu an {fiyat:.2f} ({degisim:+.1f}%)")
            mesaj = "🗓️ GÜN SONU AÇIK POZİSYONLAR\n\n" + "\n".join(satirlar)
        else:
            mesaj = "🗓️ Gün sonu: açık pozisyon yok."
        telegram_gonder(mesaj)


if __name__ == "__main__":
    main()
