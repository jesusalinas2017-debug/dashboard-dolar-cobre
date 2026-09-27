# =============================================================================
#  Dashboard Ejecutivo: Dólar y Cobre
#  Ayudantía - IA para la Gestión Industrial (UDD)
#  Se construye en 8 pasos; cada paso agrega una parte al final de este archivo.
#  Ejecutar:  streamlit run app.py
# =============================================================================

# ============================ PASO 1: BASE ====================================
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller

warnings.filterwarnings("ignore")  # statsmodels avisa mucho al ajustar; no son errores

# -----------------------------------------------------------------------------
# 1. Configuración visual (paleta ejecutiva)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Dólar y Cobre", layout="wide")

TINTA = "#1B2430"      # texto principal
TENUE = "#5F6B7A"      # texto secundario
GRILLA = "#E6E9EE"     # líneas de grilla
PANEL = "#F5F7FA"      # fondo de tarjetas
AZUL = "#1F5A96"       # dólar / modelo 1
COBRE = "#C0692B"      # cobre / modelo 2
VERDE = "#1B8A6B"      # modelo 3
GRIS = "#8A94A3"       # referencia (paseo aleatorio)
SUBE = "#1F5A96"       # barras de variación positiva
BAJA = "#C0492B"       # barras de variación negativa

# Datos de presentación de cada serie
META = {
    "usd_clp": {"nombre": "Dólar observado", "unidad": "CLP por USD", "dec": 1, "color": AZUL,
                "banda": "rgba(31,90,150,0.15)"},
    "cobre": {"nombre": "Cobre COMEX", "unidad": "US$ por libra", "dec": 3, "color": COBRE,
              "banda": "rgba(192,105,43,0.15)"},
}
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]

# Colores por modelo (fijos: el color sigue al modelo, no a su ranking)
MODELOS = {
    "ARIMA(1,1,1)": {"color": AZUL, "linea": "solid"},
    "SARIMA(1,1,1)(1,1,1,12)": {"color": COBRE, "linea": "dash"},
    "Holt-Winters": {"color": VERDE, "linea": "dashdot"},
    "Paseo aleatorio": {"color": GRIS, "linea": "dot"},
}

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.4rem; max-width: 1350px; }}
  .encabezado {{ border-left: 5px solid {AZUL}; padding: 4px 0 4px 16px; margin-bottom: 8px; }}
  .encabezado h1 {{ font-size: 1.9rem; margin: 0; color: {TINTA}; }}
  .encabezado p {{ margin: 2px 0 0; color: {TENUE}; font-size: .95rem; }}
  [data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid {GRILLA};
      border-radius: 8px; padding: 12px 16px; }}
  [data-testid="stMetricLabel"] p {{ color: {TENUE}; font-size: .85rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.7rem; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 2px; border-bottom: 1px solid {GRILLA}; }}
  .stTabs [data-baseweb="tab"] {{ padding: 8px 16px; }}
  .nota {{ color: {TENUE}; font-size: .9rem; }}
  .caja {{ background: {PANEL}; border: 1px solid {GRILLA}; border-radius: 8px;
      padding: 14px 18px; margin-bottom: 10px; }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Funciones de apoyo
# -----------------------------------------------------------------------------
def num(x, dec=1):
    """Formato chileno: punto para miles y coma para decimales."""
    texto = f"{x:,.{dec}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def mes_texto(fecha):
    return f"{MESES[fecha.month - 1]} {fecha.year}"


def estilo(fig, alto=380, leyenda=True, titulo=None):
    """Formato común para todos los gráficos (título arriba, leyenda abajo si hay título)."""
    fig.update_layout(
        height=alto, template="plotly_white", separators=",.",
        font=dict(color=TINTA, size=13), hovermode="x unified",
        margin=dict(l=10, r=10, t=50 if titulo else 30, b=70 if (titulo and leyenda) else 10),
        showlegend=leyenda,
        legend=dict(orientation="h", x=0, title=None,
                    **({"yanchor": "top", "y": -0.18} if titulo else {"yanchor": "bottom", "y": 1.02})),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    if titulo:
        fig.update_layout(title=dict(text=titulo, font=dict(size=15), x=0, xanchor="left"))
    fig.update_xaxes(gridcolor=GRILLA, zeroline=False, title=None)
    fig.update_yaxes(gridcolor=GRILLA, zeroline=False)
    return fig


VERSION_ST = tuple(int(x) for x in st.__version__.split(".")[:2])


def grafico(fig, lugar=None, key=None, barra=False):
    """Muestra un gráfico Plotly usando todo el ancho disponible."""
    lugar = lugar or st
    config = {"displayModeBar": barra, "displaylogo": False}
    if VERSION_ST >= (1, 51):  # desde 1.51 se usa width="stretch"
        lugar.plotly_chart(fig, width="stretch", config=config, key=key)
    else:
        lugar.plotly_chart(fig, use_container_width=True, config=config, key=key)


def meta_de(col):
    """Metadatos de una serie (si viene de un CSV propio, se generan)."""
    return META.get(col, {"nombre": col, "unidad": "", "dec": 2, "color": AZUL,
                          "banda": "rgba(31,90,150,0.15)"})


def metricas(real, pred):
    error = real - pred
    return {"MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAPE (%)": float(np.mean(np.abs(error / real)) * 100)}


# -----------------------------------------------------------------------------
# 3. Datos
# -----------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv("series_clase.csv", parse_dates=["fecha"])
    return df.set_index("fecha").resample("MS").mean()  # frecuencia mensual

with st.sidebar:
    st.markdown("### Panel de control")
    archivo = st.file_uploader("Cargar otra base (CSV con columna 'fecha')", type="csv")

if archivo is not None:
    df = pd.read_csv(archivo, parse_dates=["fecha"]).set_index("fecha")
    df = df.select_dtypes("number").resample("MS").mean().dropna()
    fuente = f"Base cargada: {archivo.name}"
else:
    df = cargar_datos()
    fuente = "Yahoo Finance (USDCLP=X y HG=F), promedio mensual de cierres diarios"

series = list(df.columns)
ultimo_mes = df.index[-1]



# Barra lateral: serie a analizar (los controles del modelo se agregan en el Paso 6)
with st.sidebar:
    serie_sel = st.selectbox("Serie a analizar", series, format_func=lambda c: meta_de(c)["nombre"])
    st.caption(f"Fuente: {fuente}")

meta = meta_de(serie_sel)
y = df[serie_sel].dropna()

# Encabezado y pestañas (se crean todas ahora y se llenan paso a paso)
st.markdown(f"""
<div class="encabezado">
  <h1>Dólar y cobre: evolución y pronóstico</h1>
  <p>Datos mensuales de {mes_texto(df.index[0])} a {mes_texto(ultimo_mes)} · {len(df)} meses ·
     Modelos ARIMA, SARIMA y Holt-Winters</p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Resumen", "Evolución", "Dólar vs cobre", "Estacionalidad",
                "Pronóstico", "Comparación de modelos", "Conclusiones"])

# ============================ PASO 2: RESUMEN =================================
with tabs[0]:
    st.subheader(f"Indicadores a {mes_texto(ultimo_mes)}")
    for col in series:
        m = meta_de(col)
        s = df[col].dropna()
        var_mes = (s.iloc[-1] / s.iloc[-2] - 1) * 100
        var_anual = (s.iloc[-1] / s.iloc[-13] - 1) * 100 if len(s) > 13 else np.nan
        prom_12 = s.iloc[-12:].mean()

        st.markdown(f"**{m['nombre']}** <span class='nota'>({m['unidad']})</span>",
                    unsafe_allow_html=True)
        izq, der = st.columns([1.25, 1])
        c1, c2 = izq.columns(2)
        c3, c4 = izq.columns(2)
        c1.metric("Último valor", num(s.iloc[-1], m["dec"]), f"{num(var_mes, 1)}% vs mes anterior",
                  delta_color="off")
        c2.metric("Variación 12 meses", f"{num(var_anual, 1)}%")
        c3.metric("Promedio 12 meses", num(prom_12, m["dec"]))
        c4.metric("Máximo 12 meses", num(s.iloc[-12:].max(), m["dec"]))

        # Mini gráfico de los últimos 36 meses con el último punto destacado
        u = s.iloc[-36:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=u.index, y=u, mode="lines", line=dict(color=m["color"], width=2),
                                 fill="tozeroy", fillcolor=m["banda"], name=m["nombre"],
                                 hovertemplate="%{y:,." + str(m["dec"]) + "f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[u.index[-1]], y=[u.iloc[-1]], mode="markers",
                                 marker=dict(color=m["color"], size=9, line=dict(color="white", width=2)),
                                 hoverinfo="skip"))
        estilo(fig, alto=215, leyenda=False)
        fig.update_layout(margin=dict(l=0, r=0, t=5, b=0))
        fig.update_yaxes(range=[u.min() * 0.97, u.max() * 1.03], showticklabels=False)
        fig.update_xaxes(showticklabels=False, showgrid=False)
        der.caption("Últimos 36 meses")
        grafico(fig, der, key=f"mini_{col}", barra=False)

    # Lectura rápida automática
    if {"usd_clp", "cobre"} <= set(series):
        var = df[["usd_clp", "cobre"]].pct_change().dropna()
        corr = var["usd_clp"].corr(var["cobre"])
        lectura_corr = ("cuando el cobre sube, el dólar tiende a bajar (Chile exporta cobre y entran más dólares)."
                        if corr < 0 else "ambas series tienden a moverse en la misma dirección.")
        var12 = {c: (df[c].iloc[-1] / df[c].iloc[-13] - 1) * 100 for c in ["usd_clp", "cobre"]}
        st.markdown(f"""
<div class="caja">
<b>Lectura rápida</b><br>
• La correlación entre las variaciones mensuales del dólar y del cobre es <b>{num(corr, 2)}</b>:
{lectura_corr}<br>
• En los últimos 12 meses el dólar cambió <b>{num(var12['usd_clp'], 1)}%</b> y el cobre
<b>{num(var12['cobre'], 1)}%</b>.
</div>""", unsafe_allow_html=True)

# ============================ PASO 3: EVOLUCIÓN ===============================
with tabs[1]:
    st.subheader("Evolución histórica")
    botones = [dict(count=6, label="6M", step="month", stepmode="backward"),
               dict(count=1, label="1A", step="year", stepmode="backward"),
               dict(count=3, label="3A", step="year", stepmode="backward"),
               dict(count=5, label="5A", step="year", stepmode="backward"),
               dict(step="all", label="Todo")]

    columnas = st.columns(len(series)) if len(series) <= 2 else [st.container() for _ in series]
    for contenedor, col in zip(columnas, series):
        m = meta_de(col)
        s = df[col].dropna()
        fig = go.Figure(go.Scatter(
            x=s.index, y=s, mode="lines", name=m["nombre"], line=dict(color=m["color"], width=2),
            customdata=[mes_texto(f) for f in s.index],
            hovertemplate="%{customdata}: %{y:,." + str(m["dec"]) + "f}<extra></extra>"))
        estilo(fig, alto=360, leyenda=False)
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(rangeselector=dict(buttons=botones, x=0, y=1.02))
        contenedor.markdown(f"**{m['nombre']}** <span class='nota'>({m['unidad']})</span>",
                            unsafe_allow_html=True)
        grafico(fig, contenedor, key=f"hist_{col}")

    # Índice base 100: compara series de distinta escala en UN solo eje
    st.markdown("**Índice base 100** (enero del primer año = 100): permite comparar el ritmo "
                "de ambas series aunque estén en unidades distintas.")
    indice = df / df.iloc[0] * 100
    fig = go.Figure()
    for col in series:
        m = meta_de(col)
        fig.add_trace(go.Scatter(x=indice.index, y=indice[col], name=m["nombre"], mode="lines",
                                 line=dict(color=m["color"], width=2),
                                 hovertemplate="%{y:,.1f}<extra>" + m["nombre"] + "</extra>"))
    fig.add_hline(y=100, line=dict(color=GRIS, width=1, dash="dot"))
    grafico(estilo(fig, alto=360), key="indice")

    # Variación mensual de la serie elegida
    st.markdown(f"**Variación mensual (%) — {meta['nombre']}**")
    var = (y.pct_change() * 100).dropna()
    fig = go.Figure(go.Bar(x=var.index, y=var, marker_color=np.where(var >= 0, SUBE, BAJA),
                           customdata=[mes_texto(f) for f in var.index],
                           hovertemplate="%{customdata}: %{y:+.2f}%<extra></extra>"))
    estilo(fig, alto=280, leyenda=False)
    fig.update_layout(hovermode="closest", bargap=0.15)
    grafico(fig, key="var_mensual")
    st.caption(f"Meses al alza: {(var >= 0).sum()} · meses a la baja: {(var < 0).sum()} · "
               f"mayor alza: {num(var.max(), 1)}% ({mes_texto(var.idxmax())}) · "
               f"mayor caída: {num(var.min(), 1)}% ({mes_texto(var.idxmin())})")

# ============================ PASO 4: DÓLAR VS COBRE ==========================
with tabs[2]:
    st.subheader("Relación entre el dólar y el cobre")
    if {"usd_clp", "cobre"} <= set(series):
        var = (df[["usd_clp", "cobre"]].pct_change() * 100).dropna()
        corr = var["usd_clp"].corr(var["cobre"])
        pendiente, intercepto = np.polyfit(var["cobre"], var["usd_clp"], 1)

        k1, k2, k3 = st.columns(3)
        k1.metric("Correlación (variaciones mensuales)", num(corr, 2))
        k2.metric("Si el cobre sube 1%, el dólar cambia", f"{num(pendiente, 2)}%")
        k3.metric("Meses analizados", len(var))

        c1, c2 = st.columns(2)
        # Dispersión con recta de tendencia
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=var["cobre"], y=var["usd_clp"], mode="markers", name="Meses",
            marker=dict(color=AZUL, size=8, opacity=0.6, line=dict(color="white", width=1)),
            customdata=[mes_texto(f) for f in var.index],
            hovertemplate="%{customdata}<br>Cobre: %{x:+.1f}%<br>Dólar: %{y:+.1f}%<extra></extra>"))
        xs = np.linspace(var["cobre"].min(), var["cobre"].max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=intercepto + pendiente * xs, mode="lines", name="Tendencia",
                                 line=dict(color=COBRE, width=2), hoverinfo="skip"))
        estilo(fig, alto=420, titulo="Variación mensual (%)")
        fig.update_layout(hovermode="closest")
        fig.update_xaxes(title="Cobre (%)", zeroline=True, zerolinecolor=GRILLA)
        fig.update_yaxes(title="Dólar (%)", zeroline=True, zerolinecolor=GRILLA)
        grafico(fig, c1, key="dispersion")

        # Correlación móvil de 12 meses: ¿la relación es estable en el tiempo?
        movil = var["usd_clp"].rolling(12).corr(var["cobre"]).dropna()
        fig = go.Figure(go.Scatter(x=movil.index, y=movil, mode="lines", name="Correlación 12M",
                                   line=dict(color=AZUL, width=2), fill="tozeroy",
                                   fillcolor="rgba(31,90,150,0.12)",
                                   customdata=[mes_texto(f) for f in movil.index],
                                   hovertemplate="%{customdata}: %{y:.2f}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=GRIS, width=1))
        estilo(fig, alto=420, leyenda=False, titulo="Correlación móvil de 12 meses")
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(range=[-1, 1])
        grafico(fig, c2, key="corr_movil")
        st.caption("Correlación no es causalidad: el dólar también depende de tasas de interés, "
                   "del dólar global y de la situación política y económica.")
    else:
        st.info("Esta pestaña necesita las columnas usd_clp y cobre.")

# ============================ PASO 5: ESTACIONALIDAD ==========================
with tabs[3]:
    st.subheader(f"Estacionalidad y diagnóstico — {meta['nombre']}")
    desc = seasonal_decompose(y, model="additive", period=12)
    peso = (desc.seasonal.max() - desc.seasonal.min()) / (y.max() - y.min()) * 100
    p_nivel = adfuller(y)[1]
    p_dif = adfuller(y.diff().dropna())[1]

    k1, k2, k3 = st.columns(3)
    k1.metric("Peso de la estacionalidad", f"{num(peso, 0)}% del rango")
    k2.metric("Prueba ADF: p-valor en nivel", num(p_nivel, 3))
    k3.metric("Prueba ADF: p-valor con 1 diferencia", num(p_dif, 3))
    estado = lambda p: "estacionaria" if p < 0.05 else "no estacionaria"
    st.caption(f"En nivel la serie es {estado(p_nivel)}; con una diferencia es {estado(p_dif)} "
               f"(se considera estacionaria si el p-valor es menor a 0,05). "
               + ("Por eso se usa d = 1 en ARIMA." if p_nivel >= 0.05 and p_dif < 0.05 else ""))

    c1, c2 = st.columns(2)
    # Tendencia
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y.index, y=y, name="Serie", mode="lines", line=dict(color=GRIS, width=1)))
    fig.add_trace(go.Scatter(x=desc.trend.index, y=desc.trend, name="Tendencia", mode="lines",
                             line=dict(color=meta["color"], width=2.5)))
    estilo(fig, alto=340, titulo="Serie y tendencia")
    grafico(fig, c1, key="tendencia")

    # Perfil estacional: variación promedio por mes del año
    var = (y.pct_change() * 100).dropna()
    perfil = var.groupby(var.index.month).mean()
    fig = go.Figure(go.Bar(x=[MESES[i - 1] for i in perfil.index], y=perfil,
                           marker_color=np.where(perfil >= 0, SUBE, BAJA),
                           hovertemplate="%{x}: %{y:+.2f}% promedio<extra></extra>"))
    estilo(fig, alto=340, leyenda=False, titulo="Variación promedio por mes del año (%)")
    fig.update_layout(hovermode="closest")
    grafico(fig, c2, key="perfil")

    # ACF con bandas de significancia
    x = y.diff().dropna()
    valores = acf(x, nlags=36)[1:]
    limite = 1.96 / np.sqrt(len(x))
    rezagos = np.arange(1, 37)
    colores = [COBRE if r in (12, 24, 36) else AZUL for r in rezagos]
    fig = go.Figure(go.Bar(x=rezagos, y=valores, marker_color=colores,
                           hovertemplate="Rezago %{x}: %{y:.2f}<extra></extra>"))
    fig.add_hrect(y0=-limite, y1=limite, fillcolor=GRILLA, opacity=0.6, line_width=0, layer="below")
    estilo(fig, alto=320, leyenda=False, titulo="Autocorrelación (ACF) de la serie diferenciada")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(dtick=6, title="Rezago (meses)")
    grafico(fig, key="acf")
    significativos = [r for r, v in zip(rezagos, valores) if abs(v) > limite and r in (12, 24, 36)]
    st.caption("Barras color cobre = rezagos 12, 24 y 36 (un año, dos y tres). La franja gris es la zona "
               "no significativa. " + ("Algún rezago anual sale de la franja: hay señal de estacionalidad."
                                       if significativos else
                                       "Ningún rezago anual sale de la franja: no hay estacionalidad anual clara."))

# ============================ PASO 6: MODELOS Y PRONÓSTICO ====================
# Funciones de los modelos (cada ajuste queda en caché: mover un control no reentrena)
def ajustar_modelo(y, nombre, orden=(1, 1, 1), orden_est=(1, 1, 1, 12)):
    """Ajusta el modelo elegido sobre la serie y lo devuelve."""
    if nombre.startswith("ARIMA"):
        return SARIMAX(y, order=orden).fit(disp=False)
    if nombre.startswith("SARIMA"):
        return SARIMAX(y, order=orden, seasonal_order=orden_est).fit(disp=False)
    if nombre == "Holt-Winters":
        return ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add",
                                    seasonal_periods=12).fit()
    raise ValueError(nombre)


def pronosticar(modelo, nombre, y, h, alpha):
    """Media e intervalo de confianza para h meses."""
    fechas = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    if nombre == "Holt-Winters":
        media = np.asarray(modelo.forecast(h))
        # banda aproximada: error de ajuste que crece con la raíz del horizonte
        z = {0.20: 1.2816, 0.10: 1.6449, 0.05: 1.96}[round(alpha, 2)]
        paso = z * np.std(modelo.resid) * np.sqrt(np.arange(1, h + 1))
        return pd.DataFrame({"media": media, "inferior": media - paso,
                             "superior": media + paso}, index=fechas)
    f = modelo.get_forecast(h).summary_frame(alpha=alpha)
    return pd.DataFrame({"media": f["mean"].values, "inferior": f["mean_ci_lower"].values,
                         "superior": f["mean_ci_upper"].values}, index=fechas)


@st.cache_data(show_spinner="Evaluando modelos en los últimos 12 meses...")
def comparar_modelos(y, n_test=12):
    """Backtest: cada modelo se entrena sin los últimos 12 meses y los pronostica."""
    train, test = y.iloc[:-n_test], y.iloc[-n_test:]
    preds, tabla = {}, []
    for nombre in MODELOS:
        if nombre == "Paseo aleatorio":
            pred = pd.Series(train.iloc[-1], index=test.index)  # "mañana = hoy"
        else:
            modelo = ajustar_modelo(train, nombre)
            pred = pd.Series(np.asarray(modelo.forecast(n_test)), index=test.index)
        preds[nombre] = pred
        tabla.append({"Modelo": nombre, **metricas(test, pred)})
    tabla = pd.DataFrame(tabla).sort_values("RMSE").reset_index(drop=True)
    return preds, tabla


@st.cache_data(show_spinner="Entrenando modelo...")
def pronostico_final(y, nombre, orden, orden_est, h, alpha):
    modelo = ajustar_modelo(y, nombre, orden, orden_est)
    return pronosticar(modelo, nombre, y, h, alpha)


# Controles del modelo en la barra lateral
with st.sidebar:
    st.markdown("### Modelo de pronóstico")
    modelo_sel = st.selectbox("Modelo", ["ARIMA", "SARIMA", "Holt-Winters"])
    horizonte = st.slider("Horizonte (meses)", 1, 24, 12)
    nivel = st.select_slider("Nivel de confianza", ["80%", "90%", "95%"], value="95%")
    alpha = {"80%": 0.20, "90%": 0.10, "95%": 0.05}[nivel]

    orden, orden_est = (1, 1, 1), (1, 1, 1, 12)
    if modelo_sel != "Holt-Winters":
        with st.expander("Parámetros del modelo"):
            p = st.slider("p (autorregresivo)", 0, 3, 1)
            d = st.slider("d (diferencias)", 0, 2, 1)
            q = st.slider("q (media móvil)", 0, 3, 1)
            orden = (p, d, q)
            if modelo_sel == "SARIMA":
                P = st.slider("P (estacional)", 0, 2, 1)
                D = st.slider("D (estacional)", 0, 1, 1)
                Q = st.slider("Q (estacional)", 0, 2, 1)
                orden_est = (P, D, Q, 12)

with tabs[4]:
    nombre_modelo = {"ARIMA": f"ARIMA{orden}", "SARIMA": f"SARIMA{orden}{orden_est}",
                     "Holt-Winters": "Holt-Winters"}[modelo_sel]
    st.subheader(f"Pronóstico de {meta['nombre']} con {nombre_modelo}")
    try:
        pron = pronostico_final(y, modelo_sel, orden, orden_est, horizonte, alpha)
    except Exception as e:
        pron = None
        st.error(f"El modelo no se pudo ajustar con esos parámetros. Prueba otros valores. Detalle: {e}")

    if pron is not None:
        ultimo = y.iloc[-1]
        final = pron["media"].iloc[-1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Último dato ({mes_texto(ultimo_mes)})", num(ultimo, meta["dec"]))
        k2.metric(f"Pronóstico a {horizonte} meses", num(final, meta["dec"]),
                  f"{num((final / ultimo - 1) * 100, 1)}%")
        k3.metric(f"Mínimo probable ({nivel})", num(pron["inferior"].iloc[-1], meta["dec"]))
        k4.metric(f"Máximo probable ({nivel})", num(pron["superior"].iloc[-1], meta["dec"]))

        hist = y.iloc[-60:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Observado", mode="lines",
                                 line=dict(color=meta["color"], width=2),
                                 hovertemplate="%{y:,." + str(meta["dec"]) + "f}<extra>Observado</extra>"))
        fig.add_trace(go.Scatter(x=pron.index, y=pron["superior"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pron.index, y=pron["inferior"], mode="lines", fill="tonexty",
                                 line=dict(width=0),
                                 fillcolor=meta["banda"], name=f"Intervalo {nivel}", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pron.index, y=pron["media"], name="Pronóstico", mode="lines+markers",
                                 line=dict(color=TINTA, width=2, dash="dash"), marker=dict(size=6),
                                 customdata=np.stack([pron["inferior"], pron["superior"]], axis=-1),
                                 hovertemplate="%{y:,." + str(meta["dec"]) + "f} [%{customdata[0]:,." +
                                               str(meta["dec"]) + "f} – %{customdata[1]:,." +
                                               str(meta["dec"]) + "f}]<extra>Pronóstico</extra>"))
        fig.add_vline(x=ultimo_mes, line=dict(color=GRIS, width=1, dash="dot"))
        grafico(estilo(fig, alto=460), key="pronostico")
        st.caption("La banda se abre con el horizonte: mientras más lejos, más incertidumbre. "
                   "Para decisiones conviene mirar el rango, no solo el valor central.")

        tabla = pron.copy()
        tabla.index = [mes_texto(f) for f in tabla.index]
        tabla.columns = ["Pronóstico", f"Mínimo ({nivel})", f"Máximo ({nivel})"]
        with st.expander("Ver tabla del pronóstico"):
            st.dataframe(tabla.round(meta["dec"]))
        st.download_button("Descargar pronóstico (CSV)", tabla.round(meta["dec"]).to_csv().encode("utf-8"),
                           file_name=f"pronostico_{serie_sel}.csv", mime="text/csv")


# ============================ PASO 7: COMPARACIÓN DE MODELOS ==================
with tabs[5]:
    st.subheader(f"¿Qué modelo pronostica mejor? — {meta['nombre']}")
    st.markdown("<span class='nota'>Cada modelo se entrenó sin los últimos 12 meses y luego se comparó "
                "su pronóstico contra lo que realmente pasó. El <b>paseo aleatorio</b> (\"el próximo mes "
                "será igual a este\") es la referencia mínima que un modelo debería superar.</span>",
                unsafe_allow_html=True)
    preds, tabla_m = comparar_modelos(y)
    ganador = tabla_m.loc[0, "Modelo"]
    ref = tabla_m.set_index("Modelo").loc["Paseo aleatorio", "RMSE"]

    k1, k2, k3 = st.columns(3)
    k1.metric("Mejor modelo (menor RMSE)", ganador)
    k2.metric("Su error promedio (MAPE)", f"{num(tabla_m.loc[0, 'MAPE (%)'], 1)}%")
    mejora = (1 - tabla_m.loc[0, "RMSE"] / ref) * 100
    k3.metric("Mejora vs paseo aleatorio",
              f"{num(mejora, 1)}%" if ganador != "Paseo aleatorio" else "Ninguno")

    c1, c2 = st.columns([1.5, 1])
    real = y.iloc[-36:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real.index, y=real, name="Real", mode="lines",
                             line=dict(color=TINTA, width=2.5)))
    for nombre, pred in preds.items():
        fig.add_trace(go.Scatter(x=pred.index, y=pred, name=nombre, mode="lines",
                                 line=dict(color=MODELOS[nombre]["color"], width=2,
                                           dash=MODELOS[nombre]["linea"])))
    estilo(fig, alto=440, titulo="Pronóstico de los últimos 12 meses vs real")
    grafico(fig, c1, key="backtest")

    orden_barras = tabla_m.sort_values("MAPE (%)", ascending=False)
    fig = go.Figure(go.Bar(
        x=orden_barras["MAPE (%)"], y=orden_barras["Modelo"], orientation="h",
        marker_color=[MODELOS[mo]["color"] for mo in orden_barras["Modelo"]],
        text=[f"{num(v, 1)}%" for v in orden_barras["MAPE (%)"]], textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
    estilo(fig, alto=440, leyenda=False, titulo="Error porcentual (MAPE)")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[0, orden_barras["MAPE (%)"].max() * 1.25], ticksuffix="%")
    grafico(fig, c2, key="barras_mape")

    mostrar = tabla_m.copy()
    for c in ["MAE", "RMSE"]:
        mostrar[c] = mostrar[c].round(meta["dec"])
    mostrar["MAPE (%)"] = mostrar["MAPE (%)"].round(2)
    st.dataframe(mostrar, hide_index=True)


# ============================ PASO 8: CONCLUSIONES ============================
with tabs[6]:
    st.subheader("Conclusiones")
    texto = []
    for col in series:
        m = meta_de(col)
        s = df[col].dropna()
        _, t = comparar_modelos(s)
        d_ = seasonal_decompose(s, model="additive", period=12)
        peso_c = (d_.seasonal.max() - d_.seasonal.min()) / (s.max() - s.min()) * 100
        var12 = (s.iloc[-1] / s.iloc[-13] - 1) * 100
        texto.append(
            f"**{m['nombre']}.** Cerró {mes_texto(s.index[-1])} en {num(s.iloc[-1], m['dec'])} "
            f"{m['unidad']} ({num(var12, 1)}% en 12 meses). "
            + (f"La estacionalidad pesa solo {num(peso_c, 0)}% de su rango, así que lo que manda es la "
               f"tendencia y los shocks del mercado. " if peso_c < 20 else
               f"La estacionalidad pesa {num(peso_c, 0)}% de su rango: hay un patrón anual que conviene "
               f"modelar con SARIMA. ") +
            f"En la prueba de 12 meses ganó **{t.loc[0, 'Modelo']}** con un error de "
            f"{num(t.loc[0, 'MAPE (%)'], 1)}%.")
    if {"usd_clp", "cobre"} <= set(series):
        var = df[["usd_clp", "cobre"]].pct_change().dropna()
        c = var["usd_clp"].corr(var["cobre"])
        texto.append(f"**Relación.** La correlación de las variaciones mensuales es {num(c, 2)}"
                     + (": un cobre al alza suele venir con un peso más fuerte (dólar a la baja). "
                        if c < 0 else ". ") + "Es asociación, no causalidad.")
    texto.append("**Sobre los modelos.** SARIMA solo aporta cuando la serie repite un patrón cada año "
                 "(ventas, demanda eléctrica, turismo). En precios de mercado como el dólar y el cobre, "
                 "modelos simples como ARIMA o incluso el paseo aleatorio suelen ser difíciles de superar.")
    texto.append("**Recomendación.** Usar el pronóstico con su banda de confianza, reentrenar cada mes "
                 "con el dato nuevo y preferir horizontes cortos (1 a 6 meses) para decidir.")
    for t in texto:
        st.markdown(t)

st.markdown(f"<hr style='border:none;border-top:1px solid {GRILLA}'>"
            f"<p class='nota'>Ayudantía IA para la Gestión Industrial · UDD · Streamlit + Plotly + statsmodels</p>",
            unsafe_allow_html=True)
