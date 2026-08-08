import os
import yfinance as yf
import requests

BOT_TOKEN = os.environ["BOT_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

hisseler = ["THYAO.IS", "GARAN.IS", "ASELS.IS", "SISE.IS", "KCHOL.IS"]

def rsi_hesapla(veri, periyot=14):
    delta = veri['Close'].diff()
    kazanc = delta.where(delta > 0, 0)
    kayip = -delta.where(delta < 0, 0)
    ort_kazanc = kazanc.rolling(periyot).mean()
    ort_kayip = kayip.rolling(periyot).mean()
    rs = ort_kazanc / ort_kayip
    return 100 - (100 / (1 + rs))

mesaj = "📊 Hisse Tarama Sonucu\n\n"
bulundu = False

for hisse in hisseler:
    try:
        veri = yf.download(hisse, period="3mo", interval="1d", progress=False)
        rsi = float(rsi_hesapla(veri).iloc[-1])
        if rsi < 30:
            mesaj += f"🟢 {hisse}: RSI {rsi:.1f} — aşırı satım bölgesi\n"
            bulundu = True
    except Exception as e:
        print(f"HATA - {hisse}: {e}")
        mesaj += f"⚠️ {hisse}: veri alınamadı ({str(e)[:100]})\n"
if not bulundu:
    mesaj += "Şu an sinyal veren hisse yok."

requests.post(
    f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
    data={"chat_id": CHAT_ID, "text": mesaj}
  )
