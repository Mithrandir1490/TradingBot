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
# TICKERS (Universo Core + Nuevos)
# =========================
TICKERS = [
    # Base original e Institucionales
    "NVDA", "MU", "META", "MSFT", "GOOG", "AMZN", "AAPL", "ASML", "TSM", "AVGO",
    "PLTR", "PANW", "XOM", "VST", "NFLX", "JNJ", "NEE", "HOOD", "CVX",
    "JPM", "SHOP", "AMD", "ORCL", "TEM", "V", "GEV",
    "AMAT", "LRCX", "UNH", "ABBV", "COST", "SLB", "CAT", "DE", "MSCI",
    # Nuevos y Crecimiento Estructural
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
    if p < sell_thr:
        return "SELL"
    if p > buy_thr:
        return "BUY"
    return "HOLD"

def ensure_log_exists(path):
    if not os.path.exists(path):
        pd.DataFrame(columns=["run_timestamp","ticker","date","p_final","signal"]).to_csv(path, index=False)

def append_log(path, row):
    ensure_log_exists(path)
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False)

def get_signal(ticker: str, cfg=BOT_CFG):
    df = yf.download(
        ticker,
        period=cfg["download_period"],
        interval="1d",
        auto_adjust=False,
        progress=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")

    df = df.rename(columns=str.title)
    # Soporte para Adj Close o Close
    target_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    dfx = df[[target_col]].dropna().copy()
    dfx.columns = ["x"]

    dfx["mn"] = dfx["x"].rolling(cfg["n_ma"]).mean()
    dfx["ret1"] = dfx["x"].pct_change()
    dfx["d"] = (dfx["x"] - dfx["mn"]) / dfx["mn"]
    dfx["s"] = np.sign(dfx["ret1"]).replace(0, np.nan)
    dfx["up_next"] = (dfx["ret1"].shift(-1) > 0).astype(int)
    dfx = dfx.dropna().copy()

    if len(dfx) < cfg["window_days"] + 5:
        raise RuntimeError(f"Pocos datos para {ticker}")

    hist = dfx.iloc[-cfg["window_days"]:]
    today = hist.iloc[-1]

    p_mr = compute_p_mr(hist, float(today["d"]), cfg["bins"], cfg["alpha"])
    p_mk = compute_p_mk(hist, float(today["s"]), cfg["alpha"])
    p_final = cfg["w"]*p_mr + (1-cfg["w"])*p_mk
    sig = signal_from_p(p_final, cfg["SELL_THR"], cfg["BUY_THR"])

    date = str(hist.index[-1].date())
    row = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "date": date,
        "p_final": round(float(p_final), 6),
        "signal": sig
    }
    return row

def run_daily(tickers):
    out = []
    for t in tickers:
        try:
            out.append(get_signal(t))
        except Exception:
            continue
    df = pd.DataFrame(out)
    if df.empty:
        return df
    for _, r in df.iterrows():
        append_log(LOG_PATH, r.to_dict())
    return df

# =========================
# PDF (METODOLOGÍA)
# =========================
def build_methodology_pdf_bytes() -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.8*inch, rightMargin=0.8*inch,
        topMargin=0.8*inch, bottomMargin=0.8*inch
    )
    styles = getSampleStyleSheet()
    title = styles["Title"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    mono = styles["Code"]
    body.leading = 14
    mono.fontName = "Courier"
    mono.fontSize = 9
    mono.leading = 11

    story = []
    story.append(Paragraph("Trading Bot Core — Metodología", title))
    story.append(Paragraph("Documento explicativo de Probabilidad Combinada", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Qué hace el bot", h2))
    story.append(Paragraph(
        "Este modelo estima la probabilidad de que el próximo cierre sea positivo basándose en "
        "dos factores: Reversión a la media (distancia a la MA20) y persistencia de Markov (dirección previa).",
        body
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Parámetros Actuales", h2))
    story.append(Preformatted(
        f"""W (ventana) = {BOT_CFG['window_days']} días
n (media móvil) = {BOT_CFG['n_ma']} días
w (peso MR) = {BOT_CFG['w']}
alpha (Laplace) = {BOT_CFG['alpha']}""",
        mono
    ))
    
    doc.build(story)
    return buf.getvalue()

# =========================
# UI (STREAMLIT)
# =========================
st.set_page_config(page_title="Bot 1: Core Signal", layout="wide")
st.title("🤖 Bot 1 — Señal Core Original")
st.caption("Filtro estadístico basado en Laplace y Cadenas de Markov.")

# PDF Metodología
pdf_bytes = build_methodology_pdf_bytes()
st.download_button(
    label="📄 Descargar Metodología PDF",
    data=pdf_bytes,
    file_name="Bot1_Metodologia.pdf",
    mime="application/pdf"
)

@st.cache_data(ttl=60*15)
def cached_run():
    return run_daily(TICKERS)

with st.sidebar:
    st.header("Umbrales")
    st.write(f"🔴 SELL < {BOT_CFG['SELL_THR']:.2f}")
    st.write(f"⚪ HOLD: {BOT_CFG['SELL_THR']:.2f} – {BOT_CFG['BUY_THR']:.2f}")
    st.write(f"🟢 BUY  > {BOT_CFG['BUY_THR']:.2f}")
    run_btn = st.button("Actualizar señales")

df_raw = cached_run()
if run_btn:
    st.cache_data.clear()
    df_raw = cached_run()

if df_raw.empty:
    st.error("No se pudieron generar señales. Verifica la conexión con Yahoo Finance.")
    st.stop()

# Formateo de Tabla
df = df_raw.rename(columns={"ticker":"Ticker","signal":"Recomendación","p_final":"Valor"})
df = df[["Ticker","Recomendación","Valor"]].copy()
df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")

# Orden: BUY -> SELL -> HOLD
buy = df[df["Recomendación"]=="BUY"].sort_values("Valor", ascending=False)
sell = df[df["Recomendación"]=="SELL"].sort_values("Valor", ascending=True)
hold = df[df["Recomendación"]=="HOLD"].sort_values("Valor", ascending=False)
df_final = pd.concat([buy, sell, hold], ignore_index=True)

# Estilos de Color
def color_row(row):
    rec = row["Recomendación"]
    if rec == "BUY": return ["", "", "color: #1B7F3A; font-weight: 700;"]
    if rec == "SELL": return ["", "", "color: #B00020; font-weight: 700;"]
    return ["", "", "color: #111111;"]

def color_rec_cell(val):
    if val == "BUY": return "color: #1B7F3A; font-weight: 800;"
    if val == "SELL": return "color: #B00020; font-weight: 800;"
    return "color: #111111; font-weight: 700;"

styled = (
