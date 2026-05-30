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
# ASIGNACIÓN DE TESORERÍA ASOCIADA AL FONDO DE VELOCIDAD
# ======================================================================
CAPITAL_ASIGNADO_BOT1 = 2235.00      # 40% del Fondo de Velocidad Global
PORCENTAJE_DESPLIEGUE_DIARIO = 0.10  # Capacidad de despliegue diario (10%)
presupuesto_diario_bot1 = CAPITAL_ASIGNADO_BOT1 * PORCENTAJE_DESPLIEGUE_DIARIO

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
# INTERFAZ VISUAL PROFESIONAL Y ASIGNACIÓN AUTOMÁTICA DE TESORERÍA
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
    st.divider()
    st.metric("Fondo Total Asignado", f"${CAPITAL_ASIGNADO_BOT1:,.2f} USD")
    st.metric("Presupuesto Despliegue Hoy (10%)", f"${presupuesto_diario_bot1:,.2f} USD")
    st.divider()
    if st.button("Actualizar señales", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

df_raw = cached_run()

if df_raw is not None and not df_raw.empty:
    df = df_raw.rename(columns={"ticker":"Ticker","signal":"Recomendación","p_final":"Valor Probabilístico"})
    df = df[["Ticker","Recomendación","Valor Probabilístico"]].copy()
    
    # Inicialización de columnas métricas de asignación
    df["Asignar presupuesto en este porcentaje"] = 0.0
    df["Monto de Compra (USD)"] = 0.0
    
    buys_idx = df["Recomendación"] == "BUY"
    n_buys = buys_idx.sum()
    
    if n_buys > 0:
        # Medición del exceso de confianza estocástica sobre el umbral mínimo de compra
        exceso_conviccion = df.loc[buys_idx, "Valor Probabilístico"] - BOT_CFG["BUY_THR"]
        sum_exceso = exceso_conviccion.sum()
        
        if sum_exceso == 0:
            df.loc[buys_idx, "Asignar presupuesto en este porcentaje"] = (1.0 / n_buys) * 100
        else:
            df.loc[buys_idx, "Asignar presupuesto en este porcentaje"] = (exceso_conviccion / sum_exceso) * 100
            
        # Distribución automática del presupuesto fijo del día ($223.50 USD)
        df.loc[buys_idx, "Monto de Compra (USD)"] = (df.loc[buys_idx, "Asignar presupuesto en este porcentaje"] / 100) * presupuesto_diario_bot1

    # Separación y priorización por tipo de señal para visualización ejecutiva
    buy = df[df["Recomendación"]=="BUY"].sort_values("Valor Probabilístico", ascending=False)
    sell = df[df["Recomendación"]=="SELL"].sort_values("Valor Probabilístico", ascending=True)
    hold = df[df["Recomendación"]=="HOLD"].sort_values("Valor Probabilístico", ascending=False)
    df_final = pd.concat([buy, sell, hold], ignore_index=True)

    # Bloque normativo de UI
    st.warning(f"⚠️ **DIRECTRIZ DE CONTROL ACTUARIAL:** Todo trade ejecutado bajo la señal de este bot DEBE ser liquidado al mercado de forma inflexible a más tardar el **Día Natural 7**. Presupuesto máximo de despliegue para hoy: ${presupuesto_diario_bot1:.2f} USD.")

    if n_buys > 0:
        st.success(f"🎯 **ÓRDENES DE COMPRA DETECTADAS:** Distribución proporcional de los ${presupuesto_diario_bot1:.2f} USD asignados para la jornada.")
    else:
        st.info(f"🔮 **FONDO RETENIDO:** Hoy no hay señales de compra activas en el Bot 1. El presupuesto diario de **${presupuesto_diario_bot1:.2f} USD** se mantiene líquido en tesorería para evitar sobre-operación.")

    # Estilos CSS condicionales de nivel institucional
    def style_rec(row):
        styles = [''] * len(row)
        val = row["Recomendación"]
        idx_rec = row.index.get_loc("Recomendación")
        idx_pct = row.index.get_loc("Asignar presupuesto en este porcentaje")
        idx_usd = row.index.get_loc("Monto de Compra (USD)")
        
        if val == "BUY":
            styles[idx_rec] = "background-color: #f0fff4; color: #1b7f3a; font-weight: 800; border-left: 4px solid #1b7f3a;"
            styles[idx_pct] = "background-color: #e6fffa; color: #004d40; font-weight: bold;"
            styles[idx_usd] = "background-color: #e6fffa; color: #004d40; font-weight: bold;"
        elif val == "SELL":
            styles[idx_rec] = "background-color: #fff5f5; color: #b00020; font-weight: 800; border-left: 4px solid #b00020;"
        return styles

    styled_df = df_final.style.apply(style_rec, axis=1).format({
        "Valor Probabilístico": "{:.4f}",
        "Asignar presupuesto en este porcentaje": "{:.1f}%",
        "Monto de Compra (USD)": "${:.2f}"
    })
    
    st.subheader(f"📊 Escalafón de Señales y Gestión de Capital Normalizada: {df_raw['date'].iloc[0]}")
    st.dataframe(styled_df, use_container_width=True, height=650, hide_index=True)
else:
    st.error("Error al obtener los datos de la API financiera.")
