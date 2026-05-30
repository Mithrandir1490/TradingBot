import os
import io
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timezone, timedelta

# ======================================================================
# CONFIGURACIÓN DEL BOT OPTIMIZADA (MÁXIMA SENSIBILIDAD Y CONTROL DE RIESGOS)
# ======================================================================
BOT_CFG = {
    "window_days": 90,
    "n_ma": 20,              # Ventana base para el cálculo del EMA exponencial
    "bins": 5,
    "w": 0.6,
    "alpha": 4,
    "SELL_THR": 0.40,
    "BUY_THR":  0.60,
    "download_period": "1y",
    "hard_cut_days": 7       # Stop Loss Temporal Máximo (Días Naturales)
}
LOG_PATH = "bot_log.csv"

# ======================================================================
# UNIVERSO SELECCIONADO QUIRÚRGICAMENTE (Filtrado del Universo Base de 238)
# ======================================================================
TICKERS = [
    # --- SEMICONDUCTORES Y HARDWARE (Fuerza Alfa del Bot) ---
    "NVDA", "AMD", "AVGO", "TSM", "MU", "MRVL", "AMAT", "LRCX", "KLAC", 
    "ADI", "NXPI", "ARM", "INTC", "QCOM", "DELL", "HPE",
    # --- CIBERSEGURIDAD Y SOFTWARE DE MOMENTUM ---
    "CRWD", "PANW", "FTNT", "ZS", "DDOG", "NET", "OKTA", "PLTR", 
    "SNOW", "NOW", "TEAM", "WDAY", "HUBS", "ORCL", "CRM",
    # --- MEGACAPS DE ALTA LIQUIDEZ (Reversión Estocástica Eficiente) ---
    "META", "MSFT", "GOOGL", "AMZN", "AAPL", "NFLX", "TSLA", "UBER", "ABNB", "SHOP",
    # --- INFRAESTRUCTURA ENERGÉTICA Y HARDWARE EXTREMO DE IA ---
    "VRT", "SMCI", "ANET", "GEV", "VST", "ETN", "PWR"
]

# ======================================================================
# MOTOR MATEMÁTICO (SUAVIZADO DE LAPLACE Y DISTRIBUCIÓN POR CUANTILES)
# ======================================================================
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

# ======================================================================
# INYECCIÓN DE SENSIBILIDAD EXPONENCIAL (REDUCCIÓN RADICAL DE LAG)
# ======================================================================
def get_signal(ticker: str, cfg=BOT_CFG):
    df = yf.download(ticker, period=cfg["download_period"], interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty: raise RuntimeError(f"No data for {ticker}")

    df = df.rename(columns=str.title)
    target_col = "Adj Close" if "Adj Close" in df.columns else "Close"

    dfx = df[[target_col]].dropna().copy()
    dfx.columns = ["x"]
    
    # MEJORA ALGORÍTMICA: EMA en lugar de SMA para atrapar el cambio marginal de tendencia sin retraso
    dfx["mn"] = dfx["x"].ewm(span=cfg["n_ma"], adjust=False).mean()
    
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

# ======================================================================
# INTERFAZ VISUAL PROFESIONAL (STREAMLIT MULTIPÁGINA COMPATIBLE)
# ======================================================================
st.set_page_config(page_title="Bot 1: Core Signal EMA", layout="wide")

st.title("🤖 Bot 1 — Señal Core Optimizada (Motor EMA & Control Inflexible)")
st.markdown("---")

@st.cache_data(ttl=900)
def cached_run():
    return run_daily(TICKERS)

with st.sidebar:
    st.header("🎛️ Parámetros de Control")
    st.markdown(f"🔴 **SELL THR:** < {BOT_CFG['SELL_THR']:.2f}")
    st.markdown(f"🟢 **BUY THR:** > {BOT_CFG['BUY_THR']:.2f}")
    st.markdown(f"⏱️ **Hard Time Cut:** {BOT_CFG['hard_cut_days']} Días Naturales")
    st.markdown("---")
    if st.button("Actualizar señales", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

df_raw = cached_run()

if df_raw is not None and not df_raw.empty:
    df = df_raw.rename(columns={"ticker":"Ticker","signal":"Recomendación","p_final":"Valor Probabilístico"})
    df = df[["Ticker","Recomendación","Valor Probabilístico"]].copy()
    
    # MEJORA INYECTADA: Inicializar columna de presupuesto en 0.0%
    df["Asignar presupuesto en este porcentaje"] = 0.0
    
    # Filtrar exclusivamente los BUYS activos del día
    buys_idx = df["Recomendación"] == "BUY"
    n_buys = buys_idx.sum()
    
    if n_buys > 0:
        # Calcular el exceso de confianza matemática por encima del umbral mínimo (BUY_THR)
        exceso_conviccion = df.loc[buys_idx, "Valor Probabilístico"] - BOT_CFG["BUY_THR"]
        sum_exceso = exceso_conviccion.sum()
        
        # En el caso extremo de que todas las señales tengan exactamente el mismo valor probabilístico base
        if sum_exceso == 0:
            df.loc[buys_idx, "Asignar presupuesto en este porcentaje"] = (1.0 / n_buys) * 100
        else:
            # Distribución proporcional normalizada: la suma total es exactamente 100%
            df.loc[buys_idx, "Asignar presupuesto en este porcentaje"] = (exceso_conviccion / sum_exceso) * 100

    # Separar y priorizar tablas para la visualización ejecutiva
    buy = df[df["Recomendación"]=="BUY"].sort_values("Valor Probabilístico", ascending=False)
    sell = df[df["Recomendación"]=="SELL"].sort_values("Valor Probabilístico", ascending=True)
    hold = df[df["Recomendación"]=="HOLD"].sort_values("Valor Probabilístico", ascending=False)
    df_final = pd.concat([buy, sell, hold], ignore_index=True)

    # Alerta de Control Actuarial en la UI
    st.warning(f"⚠️ **DIRECTRIZ DE CONTROL ACTUARIAL:** Todo trade ejecutado bajo la señal de este bot DEBE ser liquidado al mercado de forma inflexible a más tardar el **Día Natural 7** ({BOT_CFG['hard_cut_days']} días desde la señal). Retener operaciones perdedoras destruye el edge predictivo del modelo (Win Rate decae de 98.6% a 58.5%).")

    def style_rec(row):
        styles = [''] * len(row)
        val = row["Recomendación"]
        idx_rec = row.index.get_loc("Recomendación")
        idx_pct = row.index.get_loc("Asignar presupuesto en este porcentaje")
        
        if val == "BUY":
            styles[idx_rec] = "background-color: #f0fff4; color: #1b7f3a; font-weight: 800; border-left: 4px solid #1b7f3a;"
            styles[idx_pct] = "background-color: #e6fffa; color: #004d40; font-weight: bold;"
        elif val == "SELL":
            styles[idx_rec] = "background-color: #fff5f5; color: #b00020; font-weight: 800; border-left: 4px solid #b00020;"
        return styles

    # Aplicar formatos específicos por columna: decimales para la Probabilidad y porcentaje para el Presupuesto
    styled_df = df_final.style.apply(style_rec, axis=1).format({
        "Valor Probabilístico": "{:.4f}",
        "Asignar presupuesto en este porcentaje": "{:.1f}%"
    })
    
    st.subheader(f"📊 Escalafón de Señales y Gestión de Capital Normalizada: {df_raw['date'].iloc[0]}")
    st.dataframe(styled_df, use_container_width=True, height=650, hide_index=True)
else:
    st.error("Error al obtener los datos de la API financiera.")
