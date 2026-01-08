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
# TICKERS (actualizados, sin ABB)
# =========================
TICKERS = [
    # Base original
    "NVDA","MU","META","MSFT","GOOG","AMZN","AAPL","ASML","TSM","AVGO",
    "PLTR","PANW","XOM","VST","NFLX","JNJ","NEE","HOOD","CVX",
    "JPM","SHOP","AMD","ORCL","TEM","V","GEV",
    "AMAT","LRCX","UNH","ABBV","COST","SLB","CAT","DE","MSCI",
    # Nuevos
    "GOOGL","BRK-B","BLK","WMT","WALMEX.MX","LLY","TSLA","CRWD","ZS","DDOG","VRT",
    "MRVL","KLAC","AVAV","NOW","FTNT","ETN","PWR","EQIX","DLR","ADI","NXPI","ROK"
]


# =========================
# MODELO
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
    if "Adj Close" not in df.columns:
        raise RuntimeError(f"No Adj Close for {ticker}")

    dfx = df[["Adj Close"]].dropna().copy()
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
            # omitimos el ticker si falla para no ensuciar la tabla del día
            continue
    df = pd.DataFrame(out)
    if df.empty:
        return df
    for _, r in df.iterrows():
        append_log(LOG_PATH, r.to_dict())
    return df


# =========================
# PDF (metodología)
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
    story.append(Paragraph("Trading Bot — Metodología y Modelo Matemático", title))
    story.append(Paragraph("Documento explicativo para usuarios y socios", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Qué hace el bot (explicación simple)", h2))
    story.append(Paragraph(
        "El bot genera una señal diaria por ticker: BUY, HOLD o SELL. "
        "La señal se basa en probabilidades estimadas usando una ventana móvil de 90 días. "
        "El objetivo es actuar solo cuando exista evidencia estadística clara.",
        body
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. Parámetros", h2))
    story.append(Preformatted(
        f"""W (ventana) = {BOT_CFG['window_days']}
n (media móvil) = {BOT_CFG['n_ma']}
bins = {BOT_CFG['bins']}
w = {BOT_CFG['w']}
alpha = {BOT_CFG['alpha']}
SELL < {BOT_CFG['SELL_THR']}, HOLD entre {BOT_CFG['SELL_THR']} y {BOT_CFG['BUY_THR']}, BUY > {BOT_CFG['BUY_THR']}""",
        mono
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Matemática (paso a paso)", h2))
    story.append(Preformatted(
r"""Sea X_t el precio (Adj Close) en el día t.

Retorno diario:
r_t = (X_t / X_{t-1}) - 1

Media móvil simple:
M_{n,t} = (1/n) * Σ_{i=0..n-1} X_{t-i}

Desviación relativa:
d_t = (X_t - M_{n,t}) / M_{n,t}

Dirección diaria:
s_t = sign(r_t) ∈ {+1, -1}

Objetivo (horizonte t→t+1):
y_{t+1} = 1 si r_{t+1} > 0
        = 0 si r_{t+1} ≤ 0""",
        mono
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("4. Probabilidades estimadas", h2))
    story.append(Preformatted(
r"""Mean Reversion (por buckets de d_t):
p_mr(t) = P(y_{t+1}=1 | d_t ∈ bucket)

Markov 1 día:
p_mk(t) = P(y_{t+1}=1 | s_t)

Suavizado Laplace:
p_smooth = (p_hat * n + 0.5*alpha) / (n + alpha)

Mezcla final:
p_final(t) = w*p_mr(t) + (1-w)*p_mk(t)""",
        mono
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("5. Regla de decisión", h2))
    story.append(Preformatted(
f"""Si p_final < {BOT_CFG['SELL_THR']:.2f}  →  SELL
Si {BOT_CFG['SELL_THR']:.2f} ≤ p_final ≤ {BOT_CFG['BUY_THR']:.2f}  →  HOLD
Si p_final > {BOT_CFG['BUY_THR']:.2f}  →  BUY""",
        mono
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph("6. Nota operativa", h2))
    story.append(Paragraph(
        "En validaciones internas, la señal funcionó mejor como filtro de entrada con captura multiday. "
        "Regla simple sugerida: si un ticker aparece como BUY, considerar mantener ~5 días. "
        "El bot no ejecuta órdenes: solo entrega señales.",
        body
    ))

    doc.build(story)
    return buf.getvalue()


# =========================
# UI (Streamlit)
# =========================
st.set_page_config(page_title="Trading Bot — Señal Diaria", layout="wide")
st.title("Trading Bot — Señal diaria")
st.caption("Tabla única: BUY arriba (verde) → SELL (rojo) → HOLD (negro). Valor = p_final.")

# PDF descargable
pdf_bytes = build_methodology_pdf_bytes()
st.download_button(
    label="📄 Descargar PDF: Metodología del Trading Bot",
    data=pdf_bytes,
    file_name="TradingBot_Metodologia.pdf",
    mime="application/pdf"
)

@st.cache_data(ttl=60*15)
def cached_run():
    return run_daily(TICKERS)

with st.sidebar:
    st.header("Actualizar")
    st.write(f"SELL < {BOT_CFG['SELL_THR']:.2f}")
    st.write(f"HOLD: {BOT_CFG['SELL_THR']:.2f} – {BOT_CFG['BUY_THR']:.2f}")
    st.write(f"BUY  > {BOT_CFG['BUY_THR']:.2f}")
    run_btn = st.button("Actualizar señales")

df_raw = cached_run()
if run_btn:
    st.cache_data.clear()
    df_raw = cached_run()

if df_raw.empty:
    st.error("No pude generar señales (posible problema temporal con datos).")
    st.stop()

# Solo 3 columnas
df = df_raw.rename(columns={"ticker":"Ticker","signal":"Recomendación","p_final":"Valor"})
df = df[["Ticker","Recomendación","Valor"]].copy()
df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")

# Orden solicitado
buy = df[df["Recomendación"]=="BUY"].sort_values("Valor", ascending=False)
sell = df[df["Recomendación"]=="SELL"].sort_values("Valor", ascending=True)
hold = df[df["Recomendación"]=="HOLD"].sort_values("Valor", ascending=False)
df_final = pd.concat([buy, sell, hold], ignore_index=True)

# Colores
def color_row(row):
    rec = row["Recomendación"]
    if rec == "BUY":
        return ["", "", "color: #1B7F3A; font-weight: 700;"]
    if rec == "SELL":
        return ["", "", "color: #B00020; font-weight: 700;"]
    return ["", "", "color: #111111;"]

def color_rec_cell(val):
    if val == "BUY":
        return "color: #1B7F3A; font-weight: 800;"
    if val == "SELL":
        return "color: #B00020; font-weight: 800;"
    return "color: #111111; font-weight: 700;"

styled = (
    df_final.style
      .format({"Valor":"{:.3f}"})
      .apply(color_row, axis=1)
      .map(color_rec_cell, subset=["Recomendación"])
)

date_str = df_raw["date"].mode().iloc[0] if "date" in df_raw.columns and not df_raw["date"].isna().all() else ""
st.subheader(f"Señales del día: {date_str}")
st.dataframe(styled, use_container_width=True, height=820)

# Descarga de bitácora
st.markdown("---")
st.subheader("Bitácora")
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, "rb") as f:
        st.download_button(
            label="⬇️ Descargar bitácora (bot_log.csv)",
            data=f,
            file_name="bot_log.csv",
            mime="text/csv"
        )
else:
    st.info("Aún no existe bitácora. Recarga la app para generar señales.")


date_str = df_raw["date"].mode().iloc[0] if "date" in df_raw.columns and not df_raw["date"].isna().all() else ""
st.subheader(f"Señales del día: {date_str}")
st.dataframe(styled, use_container_width=True, height=820)
st.caption("Bitácora: bot_log.csv (en el servidor).")
