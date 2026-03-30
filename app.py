import os
import io
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timezone

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted

# =========================
# CONFIG BOT
# =========================
BOT_CFG = {
    "window_days": 90,
    "n_ma": 20,
    "bins": 5,
    "w": 0.6,
    "alpha": 4,
    "SELL_THR": 0.40,
    "BUY_THR":  0.60,
    "download_period": "1y",
}
LOG_PATH = "bot_log.csv"

# =========================
# TICKERS (Universo Core Expandido)
# =========================
TICKERS = [
    "NVDA", "MU", "META", "MSFT", "GOOG", "AMZN", "AAPL", "ASML", "TSM", "AVGO",
    "PLTR", "PANW", "XOM", "VST", "NFLX", "JNJ", "NEE", "HOOD", "CVX",
    "JPM", "SHOP", "AMD", "ORCL", "TEM", "V", "GEV",
    "AMAT", "LRCX", "UNH", "ABBV", "COST", "SLB", "CAT", "DE", "MSCI",
    "GOOGL", "BRK-B", "BLK", "WMT", "WALMEX.MX", "LLY", "TSLA", "CRWD", "ZS", 
    "DDOG", "VRT", "MRVL", "KLAC", "AVAV", "NOW", "FTNT", "ETN", "PWR", 
    "EQIX", "DLR", "ADI", "NXPI", "ROK"
]

# =========================
# MODELO MATEMÁTICO
# =========================
def laplace_smooth(p_hat, n, alpha):
    return (p_hat*n + 0.5*alpha) / (n + alpha)

def compute_p_mr(window_hist, d_t, bins, alpha):
    dvals = window_hist["d"].dropna().values
    if len(dvals) < max(30, bins*6):
        base = window_hist["up_next"].mean()
        return laplace_smooth(base, len(window_hist), alpha)

    edges = np.unique(np.quantile(dvals, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        base = window_hist["up_next"].mean()
        return laplace_smooth(base, len(window_hist), alpha)

    bucket = pd.cut([d_t], bins=edges, include_lowest=True)[0]
    tmp = window_hist.copy()
    tmp["d_bucket"] = pd.cut(tmp["d"], bins=edges, include_lowest=True)
    g = tmp.groupby("d_bucket", observed=True)["up_next"].agg(["mean", "count"])

    if bucket not in g.index or pd.isna(bucket):
        base = tmp["up_next"].mean()
        return laplace_smooth(base, len(tmp), alpha)

    p_hat = float(g.loc[bucket, "mean"])
    n = int(g.loc[bucket, "count"])
    return laplace_smooth(p_hat, n, alpha)

def compute_p_mk(window_hist, s_t, alpha):
    g = window_hist.groupby("s")["up_next"].agg(["mean", "count"])
    if s_t not in g.index or pd.isna(s_t):
        base = window_hist["up_next"].mean()
        return laplace_smooth(base, len(window_hist), alpha)

    p_hat = float(g.loc[s_t, "mean"])
    n = int(g.loc[s_t, "count"])
    return laplace_smooth(p_hat, n, alpha)

def signal_from_p(p, sell_thr, buy_thr):
    if p < sell_thr: return "SELL"
    if p > buy_thr: return "BUY"
    return "HOLD"

def ensure_log_exists(path):
    if not os.path.exists(path):
        pd.DataFrame(columns=["run_timestamp","ticker","date","p_final","signal"]).to_csv(path, index=False)

def append_log(path, row):
    ensure_log_exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)

def get_signal(ticker: str, cfg=BOT_CFG):
    df = yf.download(ticker, period=cfg["download_period"], interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty: raise RuntimeError(f"No data for {ticker}")

    df = df.rename(columns=str.title)
    target_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    dfx = df[[target_col]].dropna().copy()
    dfx.columns = ["x"]
    dfx["mn"] = dfx["x"].rolling(cfg["n_ma"]).mean()
    dfx["ret1"] = dfx["x"].pct_change()
    dfx["d"] = (dfx["x"] - dfx["mn"]) / dfx["mn"]
    dfx["s"] = np.sign(dfx["ret1"]).replace(0, np.nan)
    dfx["up_next"] = (dfx["ret1"].shift(-1) > 0).astype(int)
    dfx = dfx.dropna().copy()

    if len(dfx) < cfg["window_days"] + 5: raise RuntimeError("Pocos datos")

    hist = dfx.iloc[-cfg["window_days"]:]
    today = hist.iloc[-1]
    p_mr = compute_p_mr(hist, float(today["d"]), cfg["bins"], cfg["alpha"])
    p_mk = compute_p_mk(hist, float(today["s"]), cfg["alpha"])
    p_final = cfg["w"]*p_mr + (1-cfg["w"])*p_mk
    
    return {
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "date": str(hist.index[-1].date()),
        "p_final": round(float(p_final), 6),
        "signal": signal_from_p(p_final, cfg["SELL_THR"], cfg["BUY_THR"])
    }

def run_daily(tickers):
    out = []
    for t in tickers:
        try: out.append(get_signal(t))
        except: continue
    df = pd.DataFrame(out)
    if not df.empty:
        for _, r in df.iterrows(): append_log(LOG_PATH, r.to_dict())
    return df

# =========================
# UI (STREAMLIT)
# =========================
st.set_page_config(page_title="Bot 1: Core Signal", layout="wide")
st.title("🤖 Bot 1 — Señal Core Original")

@st.cache_data(ttl=900)
def cached_run():
    return run_daily(TICKERS)

with st.sidebar:
    st.header("Parámetros")
    st.write(f"🔴 SELL < {BOT_CFG['SELL_THR']}")
    st.write(f"🟢 BUY > {BOT_CFG['BUY_THR']}")
    if st.button("Actualizar señales"):
        st.cache_data.clear()
        st.rerun()

df_raw = cached_run()

if not df_raw.empty:
    df = df_raw.rename(columns={"ticker":"Ticker","signal":"Recomendación","p_final":"Valor"})
    df = df[["Ticker","Recomendación","Valor"]].copy()
    
    # Ordenar Tablas
    buy = df[df["Recomendación"]=="BUY"].sort_values("Valor", ascending=False)
    sell = df[df["Recomendación"]=="SELL"].sort_values("Valor", ascending=True)
    hold = df[df["Recomendación"]=="HOLD"].sort_values("Valor", ascending=False)
    df_final = pd.concat([buy, sell, hold], ignore_index=True)

    # Estilos (Aquí estaba el error del paréntesis)
    def style_rec(val):
        if val == "BUY": return "color: #1B7F3A; font-weight: 800;"
        if val == "SELL": return "color: #B00020; font-weight: 800;"
        return "color: #111111;"

    styled_df = df_final.style.map(style_rec, subset=["Recomendación"]).format({"Valor": "{:.3f}"})
    
    st.subheader(f"Señales del día: {df_raw['date'].iloc[0]}")
    st.dataframe(styled_df, use_container_width=True, height=600)
else:
    st.error("Error al obtener datos.")
