import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# Configuración base
# =========================
st.set_page_config(page_title="Análisis Financiero - OK-DTA V2", layout="wide")
import base64
from pathlib import Path

# --- Logo -> data URI (evita problemas de ruta) ---
def data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""  # si no está, ocultamos el <img>
    b64 = base64.b64encode(p.read_bytes()).decode()
    ext = p.suffix.lower().replace(".", "") or "png"
    return f"data:image/{ext};base64,{b64}"

LOGO_URI = data_uri("Logo_balmaceda.png")  # el archivo está en la misma carpeta del .py

# Tamaño del logo (desktop)
LOGO_SIZE = 210  # px  (antes ~42)

st.markdown(f"""
<style>
.header-title{{display:flex;align-items:center;gap:16px}}
.header-title h1{{margin:0;padding:0}}
/* Logo 5× en desktop, con límites para que no rompa el layout */
.header-title img{{
  width:{LOGO_SIZE}px; 
  height:{LOGO_SIZE}px; 
  object-fit:contain; 
  border-radius:12px;
  max-width:25vw;               /* evita que ocupe demasiado ancho */
}}
/* Responsive: reduce en pantallas medianas y móviles */
@media (max-width: 1200px) {{
  .header-title img{{ width:140px; height:140px; }}
}}
@media (max-width: 640px) {{
  .header-title img{{ width:72px; height:72px; }}
}}
</style>
<div class="header-title">
  <h1>🔎 Análisis Financiero de Bodegas</h1>
  {'<img src="'+LOGO_URI+'" alt="Logo">' if LOGO_URI else ''}
</div>
""", unsafe_allow_html=True)


st.caption("Fuente: Google Sheets (CSV) · Agrupaciones dinámicas y visualizaciones interactivas")


CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSuoX_V5rYls-pBu7F3_VP2APS3FL7-eYbn9uDWUGJQZbxNfQTm9gRlyDlE69wWJjsDQpDzi2lt31Ak/pub?gid=1154929321&single=true&output=csv"

# Sidebar
st.sidebar.header("⚙️ Controles")
if st.sidebar.button("🔄 Actualizar datos (limpiar caché)"):
    st.cache_data.clear()

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # Normalización y tipos
    df["Fecha"] = pd.to_datetime(df.get("Fecha"), errors="coerce")
    df["Monto"] = pd.to_numeric(df["Monto"].astype(str).str.replace(r"[^\d\.-]", "", regex=True), errors="coerce")
    # Filtrado mínimo
    df = df.dropna(subset=["Monto", "CC"])
    # Columnas esperadas que podrían no existir
    for c in ["Obs", "CC1", "Sit", "Responsable", "Año"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df

df = load_data(CSV_URL)

# =========================
# KPIs con diseño personalizado (antes de filtros)
# =========================
st.markdown("""
    <style>
    .kpi-card {
        background-color: #000000;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-family: Arial, sans-serif;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.4);
    }
    .kpi-title {
        font-size: 16px;
        color: #bbbbbb;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .kpi-sub {
        font-size: 14px;
        color: #888888;
    }
    </style>
""", unsafe_allow_html=True)

CAPEX = 151_834_571

# Canon arriendo (se mantiene igual a tu lógica)
mask_canon = (
    (df["CC"] == "INGRESO") &
    (
        df["CC1"].astype(str).str.contains("arriendo", case=False, na=False) |
        df["Obs"].astype(str).str.contains("canon", case=False, na=False)
    )
)
# (Compatibilidad por si tu pandas no soporta el hasattr que tenías)
if not hasattr(pd.Series.str, "contains"):
    mask_canon = (
        (df["CC"] == "INGRESO") &
        (
            df["CC1"].astype(str).str.contains("arriendo", case=False, na=False) |
            df["Obs"].astype(str).str.contains("canon", case=False, na=False)
        )
    )

ingresos_canon = df.loc[mask_canon, "Monto"].sum()
cobertura_capex = ingresos_canon / CAPEX if CAPEX else 0

# NUEVO: Saldo cuenta = PAGADO + ABONOS (abono__)
total_pagado = df.loc[df["Sit"].astype(str).str.upper() == "PAGADO", "Monto"].sum()
mask_abono = df["Obs"].astype(str).str.contains(r"\babono_*\b", case=False, na=False)
total_abonos = df.loc[mask_abono, "Monto"].sum()
saldo_cuenta = total_pagado + total_abonos

# Color dinámico para el saldo (rojo si negativo, azul si positivo/cero)
saldo_color = "#EF4444" if saldo_cuenta < 0 else "#2563EB"

# 3 tarjetas centradas (espaciadores a izquierda y derecha)
sp_left, kpi1, kpi2, kpi3, sp_right = st.columns([1, 3, 3, 3, 1])

with kpi1:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">💼 CAPEX</div>
            <div class="kpi-value">${CAPEX:,.0f}</div>
            <div class="kpi-sub">Inversión total</div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">🏢 Ingresos Canon Arriendo</div>
            <div class="kpi-value">${ingresos_canon:,.0f}</div>
            <div class="kpi-sub">Acumulado</div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📊 Cobertura CAPEX</div>
            <div class="kpi-value">{(ingresos_canon / CAPEX if CAPEX else 0):.1%}</div>
            <div class="kpi-sub">Canon / CAPEX</div>
        </div>
    """, unsafe_allow_html=True)




# =========================
# SIN FILTROS: usar todo el dataset
# =========================
df_f = df.copy()

# --- Cálculo robusto para los KPI (NO pisar estos nombres en otra parte) ---
mask_ingreso = df_f["CC"].astype(str).str.upper().eq("INGRESO")
mask_egreso  = df_f["CC"].astype(str).str.upper().eq("EGRESO")

# Normaliza 'Sit' y evita que "NO PAGADO" cuente como "PAGADO"
sit_up = df_f["Sit"].astype(str).str.strip().str.upper()

mask_sit_pagado = sit_up.eq("PAGADO")           # SOLO "PAGADO"
mask_sit_abono  = sit_up.str.startswith("ABONO") # "ABONO", "ABONO__", etc.

ingresos_kpi = df_f.loc[mask_ingreso & (mask_sit_pagado | mask_sit_abono), "Monto"].sum()
egresos_kpi  = df_f.loc[mask_egreso  &  mask_sit_pagado, "Monto"].sum()

# Si tu “Balance Neto” es el saldo cuenta (PAGADO + ABONO__), usa saldo_cuenta:
balance_kpi = saldo_cuenta
# Si prefieres ingresos+egresos, usa:
# balance_kpi = ingresos_kpi + egresos_kpi


st.markdown("---")

# === KPIs financieros PRO (cards con acento por color) ===
st.markdown("""
<style>
.kpi3-card{background:#ffffff;border-radius:14px;padding:18px 20px;border:1px solid #E5E7EB;
           box-shadow:0 8px 20px rgba(2,6,23,.06);border-left:6px solid var(--accent);}
.kpi3-title{font-size:13px;color:#111827;font-weight:700;letter-spacing:.3px;text-transform:uppercase;margin-bottom:6px;}
.kpi3-value{font-variant-numeric:tabular-nums;font-size:42px;line-height:1.1;font-weight:800;margin:0;}
.kpi3-sub{display:none !important;}  /* ocultar detalle bajo el número */
</style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
bal_color = "#10B981" if balance_kpi >= 0 else "#EF4444"

with c1:
    st.markdown(
        f"""
        <div class="kpi3-card" style="--accent:#10B981">
            <div class="kpi3-title">TOTAL INGRESOS</div>
            <div class="kpi3-value" style="color:#10B981;">${ingresos_kpi:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="kpi3-card" style="--accent:#EF4444">
            <div class="kpi3-title">TOTAL EGRESOS</div>
            <div class="kpi3-value" style="color:#EF4444;">${egresos_kpi:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi3-card" style="--accent:{bal_color}">
            <div class="kpi3-title">CAJA BANCO BCI</div>
            <div class="kpi3-value" style="color:{bal_color};">${balance_kpi:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
# === FILA: Cuentas por Cobrar (neto) + Egresos por Pagar ===
sit_norm = df_f["Sit"].astype(str).str.strip().str.upper()
cc_up    = df_f["CC"].astype(str).str.upper()

# Cuentas por cobrar (neto) = NO PAGADO – ABONO  [solo INGRESO]
mask_ingreso     = cc_up.eq("INGRESO")
no_pagado_total  = df_f.loc[mask_ingreso & sit_norm.eq("NO PAGADO"), "Monto"].sum()
abonos_total     = df_f.loc[mask_ingreso & sit_norm.str.startswith("ABONO"), "Monto"].sum()
if abonos_total == 0:
    abonos_total = df_f.loc[
        mask_ingreso & df_f["Obs"].astype(str).str.contains(r"\babono\b", case=False, na=False),
        "Monto"
    ].sum()
cuentas_por_cobrar_neto = no_pagado_total - abonos_total

# Egresos por pagar [EGRESO y NO PAGADO]
total_egresos_por_pagar = df_f.loc[
    cc_up.eq("EGRESO") & sit_norm.eq("NO PAGADO"), "Monto"
].sum()

# Nueva: Posición neta integral = CxC neto + EPP + Balance Neto
posicion_neta = cuentas_por_cobrar_neto + total_egresos_por_pagar + balance_kpi


# -------- Layout en una sola fila --------
c1, c2, c3 = st.columns([1,1,1])  # usa c3 si quieres mostrar la posición neta

cxc_color = "#F59E0B" if cuentas_por_cobrar_neto > 0 else "#10B981"
with c1:
    st.markdown(
        f"""
        <div class="kpi3-card" style="--accent:{cxc_color}">
            <div class="kpi3-title">CUENTAS POR COBRAR </div>
            <div class="kpi3-value" style="color:{cxc_color};">${cuentas_por_cobrar_neto:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

epp_color = "#EF4444" if total_egresos_por_pagar != 0 else "#10B981"
with c2:
    st.markdown(
        f"""
        <div class="kpi3-card" style="--accent:{epp_color}">
            <div class="kpi3-title">EGRESOS POR PAGAR</div>
            <div class="kpi3-value" style="color:{epp_color};">${total_egresos_por_pagar:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Si NO quieres mostrar "Posición Neta", borra este bloque 'with c3'
pn_color = "#10B981" if posicion_neta >= 0 else "#EF4444"
with c3:
    st.markdown(
        f"""
        <div class="kpi3-card" style="--accent:{pn_color}">
            <div class="kpi3-title">POSICIÓN NETA (CxC + EPP + BN)</div>
            <div class="kpi3-value" style="color:{pn_color};">${posicion_neta:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )



# =========================
# Gráfico PRO: Ingresos por canon de arriendo por año + Promedio móvil (MA-3)
# =========================
import plotly.graph_objects as go
import pandas as pd

st.markdown("### 🏢 Ingresos por Canon de Arriendo — por Año (con MA-3)")

# Usa df_f si existen filtros aplicados; si no, usa df completo
data_src = df_f if "df_f" in locals() else df

# Filtro canon: CC == INGRESO y (CC1 contiene "arriendo" o Obs contiene "canon")
mask_canon = (
    (data_src["CC"] == "INGRESO") &
    (
        data_src["CC1"].astype(str).str.contains("arriendo", case=False, na=False) |
        data_src["Obs"].astype(str).str.contains("canon", case=False, na=False)
    )
)

canon_year = (
    data_src.loc[mask_canon, ["Año", "Monto"]]
    .dropna(subset=["Año"])
    .groupby("Año", as_index=False)["Monto"].sum()
    .sort_values("Año")
)

# Asegurar tipo numérico del año y ordenar
canon_year["Año"] = pd.to_numeric(canon_year["Año"], errors="coerce")
canon_year = canon_year.dropna(subset=["Año"]).sort_values("Año")

# Promedio móvil simple de 3 años (min_periods=1 para no perder primeros años)
canon_year["MA3"] = (
    canon_year.sort_values("Año")["Monto"].rolling(window=3, min_periods=1).mean()
)

# Etiquetas abreviadas (k/M)
def fmt_short(v: float) -> str:
    v = float(v)
    av = abs(v)
    if av >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if av >= 1_000:
        return f"${v/1_000:.0f}k"
    return f"${v:,.0f}"

labels_bar = canon_year["Monto"].apply(fmt_short)
labels_line = canon_year["MA3"].apply(fmt_short)

# Figura PRO: Barras + Línea MA3
fig = go.Figure()

# Barras: Canon anual
fig.add_trace(go.Bar(
    x=canon_year["Año"],
    y=canon_year["Monto"],
    name="Canon anual (CLP)",
    text=labels_bar,
    textposition="outside",
    offsetgroup=1,
    hovertemplate=(
        "<b>Año %{x}</b><br>" \
        "Canon: $%{y:,.0f}<extra></extra>"
    ),
    marker=dict(
        line=dict(width=1, color="rgba(0,0,0,0.25)")
    )
))

# Línea: MA-3
fig.add_trace(go.Scatter(
    x=canon_year["Año"],
    y=canon_year["MA3"],
    name="Promedio móvil (MA-3)",
    mode="lines+markers+text",
    text=labels_line,
    textposition="top center",
    textfont=dict(color="darkblue", size=12),   # 👈 color azul oscuro y tamaño legible
    line=dict(width=3, dash="solid"),
    marker=dict(size=7),
    hovertemplate=(
        "<b>Año %{x}</b><br>MA-3: $%{y:,.0f}<extra></extra>"
    )
))


# Anotación del último valor (callout)
if not canon_year.empty:
    last_row = canon_year.iloc[-1]
    fig.add_annotation(
        x=last_row["Año"], y=last_row["Monto"],
        text=f"Último año: {int(last_row['Año'])}<br><b>{fmt_short(last_row['Monto'])}</b>",
        showarrow=True, arrowhead=2, ax=40, ay=-40,
        bgcolor="rgba(0,0,0,0.7)", bordercolor="rgba(0,0,0,0.7)", font=dict(color="white")
    )

fig.update_layout(
    title=dict(
        text="Ingresos por Canon de Arriendo — Evolución Anual",
        x=0.02, xanchor="left"
    ),
    xaxis_title="Año",
    yaxis_title="Monto (CLP)",
    yaxis_tickformat=",.0f",
    bargap=0.18,
    hovermode="x unified",
    margin=dict(l=20, r=20, t=60, b=20),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# Grid sutil PRO
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)

st.plotly_chart(fig, use_container_width=True, config={
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["toImage", "drawline", "drawrect", "eraseshape"],
})

# (Opcional) Descargar dataset agregado
csv_canon = canon_year.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar ingresos por canon (CSV)",
    data=csv_canon,
    file_name="ingresos_canon_por_anio.csv",
    mime="text/csv"
)

# =========================
# Top 10 dinámico PRO (Barras / Treemap)
# =========================
st.markdown("### 📈 Filtro por centro de costo")
c_top1, c_top2, c_top3, c_top4 = st.columns([2,1,1,1])
with c_top1:
    dim = st.selectbox("Dimensión", ["Obs", "CC1", "Sit", "CC"], index=0, key="dim_pro")
with c_top2:
    order_by = st.radio("Ordenar por", ["Total CLP", "N° Transacciones"], horizontal=True, index=0, key="order_by_pro")
with c_top3:
    chart_type = st.selectbox("Visualización", ["Barras", "Treemap"], index=0, key="chart_type_pro")
with c_top4:
    top_n = st.slider("Top N", min_value=5, max_value=30, value=16, step=1, key="topn_pro")

topN_raw = (
    df_f.groupby(dim)["Monto"]
    .agg(["sum", "count"])
    .rename(columns={"sum": "Total CLP", "count": "N° Transacciones"})
    .reset_index()
)

sort_col = "Total CLP" if order_by == "Total CLP" else "N° Transacciones"
topN = topN_raw.sort_values(sort_col, ascending=False).head(top_n).copy()

def fmt_short(v):
    av = abs(v)
    if av >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if av >= 1_000:
        return f"${v/1_000:.0f}k"
    return f"${v:,.0f}"

def color_by_sign_or_cc(series_values, series_dim=None):
    if series_dim == "CC":
        return series_values.map({"INGRESO": "#10B981", "EGRESO": "#EF4444"}).fillna("#2563EB")
    return series_values.apply(lambda v: "#10B981" if v >= 0 else "#EF4444")

if chart_type == "Barras":
    topN = topN.sort_values("Total CLP", ascending=True)
    topN["Etiqueta"] = topN["Total CLP"].apply(fmt_short)
    denom = topN["Total CLP"].sum() if order_by == "Total CLP" else topN["N° Transacciones"].sum()
    topN["% del total"] = (topN[sort_col] / (denom if denom != 0 else 1))
    bar_colors = color_by_sign_or_cc(topN[dim] if dim == "CC" else topN["Total CLP"], series_dim=dim)

    fig_top = px.bar(
        topN, x="Total CLP", y=dim, orientation="h",
        title=f"Top {top_n} por '{dim}' · {order_by}"
    )
    fig_top.update_traces(
        marker_color=bar_colors,
        marker_line_color="rgba(0,0,0,0.15)", marker_line_width=1.2,
        text=topN["Etiqueta"], texttemplate="%{text}", textposition="inside",
        insidetextanchor="middle",
        customdata=topN[["N° Transacciones", "% del total"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Total: $%{x:,.0f}<br>"
            "Transacciones: %{customdata[0]:,}<br>"
            "Participación: %{customdata[1]:.1%}<extra></extra>"
        ),
        cliponaxis=False,
    )
    if (topN["Total CLP"] < 0).any() and (topN["Total CLP"] > 0).any():
        fig_top.add_vline(x=0, line_width=1, line_dash="dash", line_color="#9CA3AF")

    fig_top.update_layout(
        xaxis_title="Total CLP", yaxis_title=dim, xaxis_tickformat=",.0f",
        bargap=0.25, margin=dict(l=20, r=20, t=60, b=20), template="plotly_white"
    )
    st.plotly_chart(fig_top, use_container_width=True,
        config={"displaylogo": False, "displayModeBar": True,
                "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape", "toImage"]})
else:
    treemap_df = topN.sort_values("Total CLP", ascending=False).copy()
    if dim == "CC":
        color_col = dim
        color_scale = None
    else:
        color_col = "Total CLP"
        color_scale = ["#EF4444", "#F59E0B", "#10B981"]
    fig_tree = px.treemap(
        treemap_df, path=[dim],
        values=sort_col if sort_col in treemap_df.columns else "Total CLP",
        color=color_col, color_continuous_scale=color_scale,
        title=f"Distribución Top {top_n} por '{dim}' · {order_by}"
    )
    if dim == "CC":
        fig_tree.update_traces(marker_colors=color_by_sign_or_cc(treemap_df[dim], series_dim="CC"))
    fig_tree.update_traces(
        hovertemplate="<b>%{label}</b><br>Valor: %{value:,.0f}<extra></extra>",
        textinfo="label+value", textfont=dict(size=12)
    )
    fig_tree.update_layout(margin=dict(l=0, r=0, t=60, b=0), template="plotly_white")
    st.plotly_chart(fig_tree, use_container_width=True,
        config={"displaylogo": False, "displayModeBar": True, "modeBarButtonsToAdd": ["toImage"]})


# =========================
# Tabla integrada: NO PAGADO + Abonos + Total (versión PRO)
# =========================
st.markdown("---")
st.header("📋 Resumen por Responsable (NO PAGADO vs Abonos)")

df_np = df_f[df_f["Sit"] == "NO PAGADO"]
df_abonos = df_f[df_f["Obs"].astype(str).str.contains("abono", case=False, na=False)]

no_pagado_grouped = df_np.groupby("Responsable")["Monto"].agg(["sum", "count"]).rename(
    columns={"sum": "Monto NO PAGADO", "count": "Transacciones NO PAGADO"}
)
abonos_grouped = df_abonos.groupby("Responsable")["Monto"].agg(["sum", "count"]).rename(
    columns={"sum": "Monto Abonos", "count": "Cantidad Abonos"}
)

resumen = no_pagado_grouped.join(abonos_grouped, how="outer").fillna(0)
resumen["Deuda"] = resumen["Monto NO PAGADO"] - resumen["Monto Abonos"]
resumen["% Abonado"] = (resumen["Monto Abonos"] / resumen["Monto NO PAGADO"]).replace([pd.NA, pd.NaT], 0).fillna(0)
resumen["Progreso"] = resumen["% Abonado"].clip(lower=0, upper=1)

def badge_pct(p):
    if p >= 1:   return "🟢 OK"
    if p >= 0.5: return "🟠 En curso"
    return "🔴 Bajo"
resumen["Estado"] = resumen["% Abonado"].apply(badge_pct)

# Ordenar por Deuda
resumen = resumen.sort_values("Deuda", ascending=False)
tabla = resumen.reset_index()

# ===== Estilo de la tabla en pantalla =====
styler = (
    tabla.style
    .format({
        "Monto NO PAGADO": "${:,.0f}",
        "Monto Abonos": "${:,.0f}",
        "Deuda": "${:,.0f}",
        "% Abonado": "{:.1%}",
        "Transacciones NO PAGADO": "{:,.0f}",
        "Cantidad Abonos": "{:,.0f}",
    })
    .hide(axis="index")
    .set_table_styles([
        {"selector": "thead th", "props": [("background-color", "#f5f6f8"), ("color", "#111827"),
                                           ("font-weight", "600"), ("font-size", "14px")]},
        {"selector": "tbody td", "props": [("font-size", "13px")]},
        {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#fbfbfc")]},
        {"selector": "tbody tr:hover", "props": [("background-color", "#f0f6ff")]}
    ])
    .background_gradient(subset=["Monto NO PAGADO"], cmap="Reds")
    .background_gradient(subset=["Monto Abonos"], cmap="Greens")
    .background_gradient(subset=["Deuda"], cmap="Oranges")
    .bar(subset=["Progreso"], color="#10B981")
)

# 👇 Letras BLANCAS SOLO en la PRIMERA FILA de la columna Deuda (fila 0 después del reset_index)
styler = styler.apply(
    lambda s: ["color: white; font-weight:700;"] + [""]*(len(s)-1),
    axis=0, subset=["Deuda"]
)

st.dataframe(styler, use_container_width=True)

# ====== Descarga en PDF (fallback a CSV si falta reportlab) ======
# ---- PDF bonito y que no se corte ----
from reportlab.lib.pagesizes import A4, A3, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.units import cm

def export_resumen_pdf(df: pd.DataFrame, filename: str = "resumen_no_pagado_abonos.pdf"):
    # 1) Renombrar cabeceras largas (opcional, ayuda mucho)
    short_names = {
        "Monto NO PAGADO": "Monto NO PAG.",
        "Transacciones NO PAGADO": "Transacc. NO PAG.",
        "Cantidad Abonos": "Cant. Abonos",
        "Monto Abonos": "Monto Abonos",
        "% Abonado": "% Abonado",
    }
    df = df.rename(columns={c: short_names.get(c, c) for c in df.columns})

    # 2) Tamaño de hoja según nº columnas
    ncols = len(df.columns)
    page_size = landscape(A4) if ncols <= 8 else landscape(A3)

    # 3) Documento con márgenes
    doc = SimpleDocTemplate(
        filename, pagesize=page_size,
        leftMargin=1*cm, rightMargin=1*cm, topMargin=1*cm, bottomMargin=1*cm
    )

    # 4) Estilos con WRAP (clave: Paragraph + wordWrap)
    base = ParagraphStyle(
        "base", fontSize=8.5, leading=10, spaceBefore=0, spaceAfter=0,
        wordWrap="CJK",  # permite cortar palabras largas
        splitLongWords=True
    )
    sty_h = ParagraphStyle("head", parent=base, alignment=TA_CENTER, fontSize=9, leading=11)
    sty_l = ParagraphStyle("left", parent=base, alignment=TA_LEFT)
    sty_r = ParagraphStyle("right", parent=base, alignment=TA_RIGHT)

    # 5) Título
    story = [
        Paragraph("<b>Resumen por Responsable (NO PAGADO vs Abonos)</b>", getSampleStyleSheet()["Title"]),
        Spacer(1, 0.4*cm)
    ]

    # 6) Preparar datos con Paragraphs para envolver texto
    #    (números a la derecha, texto a la izquierda)
       # 6) Preparar datos con Paragraphs para envolver texto
    text_cols = {"Responsable", "Estado"}
    header = [Paragraph(col, sty_h) for col in df.columns]
    rows = []
    for _, r in df.iterrows():
        row = []
        for col, val in zip(df.columns, r):
            # --- Formato de números ---
            if pd.isna(val):
                s = ""
            elif col in {"% Abonado"}:
                s = f"{float(val):,.1f} %"  # una decimal + símbolo %
            elif isinstance(val, (int, float)) and col not in text_cols:
                s = f"{val:,.0f}"           # separador de miles, sin decimales
            else:
                s = str(val)
            # --- Paragraph alineado ---
            row.append(Paragraph(s, sty_l if col in text_cols else sty_r))
        rows.append(row)
    data = [header] + rows


    # 7) Anchos de columna encasillados
    #    - Responsable ancho grande
    #    - Conteos/Estado angostos
    #    - Dinero/porcentajes medios
    page_width = doc.width
    weights = []
    for c in df.columns:
        if c.lower().startswith("responsable"):
            weights.append(3.8)      # más ancho
        elif "Transacc" in c or "Cant." in c or c == "Estado":
            weights.append(1.2)      # angosto
        else:
            weights.append(2.0)      # medio
    total = sum(weights)
    col_widths = [max(1.6*cm, page_width*(w/total)) for w in weights]  # mínimo 1.6 cm

    # 8) Tabla
    tbl = Table(data, colWidths=col_widths, repeatRows=1)

    # 9) Estilos de tabla (alineación ya la maneja Paragraph)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        # rayas cebra
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.Color(0.98,0.98,0.98), colors.white]),
    ]))

    story.append(tbl)
    doc.build(story)

# ---- Llamada desde tu app ----
try:
    pdf_path = "resumen_no_pagado_abonos.pdf"
    export_resumen_pdf(tabla, pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button("⬇️ Descargar PDF", data=f.read(), file_name=pdf_path, mime="application/pdf")
except Exception as e:
    st.warning(f"No se pudo generar PDF. Te dejo el CSV como alternativa. Detalle: {e}")
    st.download_button("⬇️ Descargar CSV", data=tabla.to_csv(index=False).encode("utf-8"),
                       file_name="resumen_no_pagado_abonos.csv", mime="text/csv")

# =========================
# Gráfico PRO: Canon Mensual por Año y por Esp (líneas)
# =========================
import plotly.graph_objects as go
import numpy as np

st.markdown("### 📈 Canon mensual por **Año** y **Esp**")

# Fuente de datos: usa df_f si quieres que respete filtros globales; usa df si lo quieres “global”
_data_src = df_f if "df_f" in locals() else df

# Filtro: sólo CANON MENSUAL (mismo criterio que definimos)
mask_canon_mensual = (
    (_data_src["CC"].astype(str) == "INGRESO") &
    (
        _data_src["CC1"].astype(str).str.contains("canon mensual", case=False, na=False) |
        _data_src["Obs"].astype(str).str.contains("canon mensual", case=False, na=False)
    )
)
dm = _data_src.loc[mask_canon_mensual].copy()

# Tipos
dm["Año"] = pd.to_numeric(dm["Año"], errors="coerce")
dm["Monto"] = pd.to_numeric(
    dm["Monto"].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
    errors="coerce"
)

# Agregado: Año x Esp
agg = (
    dm.groupby(["Año","Esp"], as_index=False)["Monto"]
      .sum()
      .dropna(subset=["Año"])
      .sort_values(["Año","Esp"])
)

# Años consecutivos (rellena con 0 si falta algún Esp en algún año)
all_years = np.arange(int(agg["Año"].min()), int(agg["Año"].max())+1) if not agg.empty else []
all_esps  = sorted(agg["Esp"].dropna().unique())
grid = pd.MultiIndex.from_product([all_years, all_esps], names=["Año","Esp"])
agg_full = (
    agg.set_index(["Año","Esp"])
       .reindex(grid, fill_value=0)
       .reset_index()
)

# Formato abreviado para etiquetas
def fmt_clp(v):
    av = abs(float(v))
    if av >= 1_000_000_000:
        return f"${v/1_000_000_000:.1f}B"
    if av >= 1_000_000:
        return f"${v/1_000_000:.1f}M"
    if av >= 1_000:
        return f"${v/1_000:.0f}k"
    return f"${v:,.0f}"

# Paleta discreta consistente (hasta 12 Esps; si hay más, Plotly cicla)
palette = [
    "#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6",
    "#14B8A6","#F97316","#DC2626","#3B82F6","#22C55E",
    "#EAB308","#EC4899"
]

# Construcción de la figura
fig_line = go.Figure()

for i, esp in enumerate(all_esps):
    df_e = agg_full[agg_full["Esp"] == esp]
    fig_line.add_trace(go.Scatter(
        x=df_e["Año"], y=df_e["Monto"],
        mode="lines+markers",
        name=f"Esp {esp}",
        line=dict(width=3, color=palette[i % len(palette)]),
        marker=dict(size=7),
        hovertemplate="<b>Año %{x}</b><br>Esp: " + str(esp) + "<br>Monto: $%{y:,.0f}<extra></extra>",
    ))

# Etiqueta del último punto de cada serie (callout)
for i, esp in enumerate(all_esps):
    df_e = agg_full[agg_full["Esp"] == esp]
    if not df_e.empty:
        last_row = df_e.iloc[-1]
        fig_line.add_annotation(
            x=last_row["Año"], y=last_row["Monto"],
            text=f"Esp {esp}<br><b>{fmt_clp(last_row['Monto'])}</b>",
            showarrow=True, arrowhead=2, ax=30, ay=-30,
            bgcolor="rgba(0,0,0,0.72)", bordercolor="rgba(0,0,0,0.72)",
            font=dict(color="white", size=11)
        )

# Botones: Mostrar/Ocultar todos
visibility_all = [True] * len(all_esps)
visibility_none = [False] * len(all_esps)
fig_line.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=1, xanchor="right", y=1.15, yanchor="top",
            buttons=[
                dict(label="Mostrar todo", method="update", args=[{"visible": visibility_all}]),
                dict(label="Ocultar todo", method="update", args=[{"visible": visibility_none}]),
            ]
        )
    ]
)

# Layout PRO
fig_line.update_layout(
    title=dict(text="Canon mensual por Año y Esp", x=0.02, xanchor="left"),
    xaxis_title="Año",
    yaxis_title="Monto (CLP)",
    hovermode="x unified",
    template="plotly_white",
    margin=dict(l=20, r=20, t=70, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.02),
)

# Ejes y grids sutiles
fig_line.update_xaxes(
    type="category" if agg_full["Año"].dtype == "O" else "linear",
    tickmode="array", tickvals=list(all_years), showgrid=False
)
fig_line.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False, tickformat=",.0f")

# Rango interactivo
fig_line.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

st.plotly_chart(fig_line, use_container_width=True, config={
    "displaylogo": False,
    "displayModeBar": True,
    "modeBarButtonsToAdd": ["toImage","drawline","drawrect","eraseshape"]
})
############################
# =========================
# 🧩 Canon por m² — SOLO "canon mensual" (por Año y Esp) • CLP/UF
# =========================
import numpy as np
import plotly.graph_objects as go

st.markdown("## 🧩 Canon por m² — Canon Mensual (por Año y Esp)")

# --- Controles (keys únicos)
c1, c2 = st.columns([1,1])
with c1:
    escala_m2 = st.radio("Escala", ["Mensual", "Diario"], horizontal=True, index=0, key="escala_m2_final")
with c2:
    uf_url = st.text_input("URL CSV UF (opcional: Año,UF_promedio)", value="", key="uf_url_m2_final",
                           placeholder="https://.../uf_promedio_anual.csv")

# --- Base de datos (respeta filtros si existen)
_df = df_f if "df_f" in locals() else df

# --- Filtro: variantes típicas de 'canon mensual'
canon_mask = (
    (_df["CC"].astype(str) == "INGRESO") &
    (
        _df["CC1"].astype(str).str.contains(r"canon\s*mensual|canon.*arriendo|arriendo.*mensual", case=False, na=False) |
        _df["Obs"].astype(str).str.contains(r"canon\s*mensual|canon.*arriendo|arriendo.*mensual", case=False, na=False)
    )
)
dm = _df.loc[canon_mask].copy()

# --- Normalización
dm["Año"] = pd.to_numeric(dm["Año"], errors="coerce")
dm["Esp"] = pd.to_numeric(dm["Esp"], errors="coerce")
dm["Monto"] = pd.to_numeric(dm["Monto"].astype(str).str.replace(r"[^\d\.-]", "", regex=True), errors="coerce")
dm = dm.dropna(subset=["Año","Esp","Monto"])
dm["Año"] = dm["Año"].astype(int)
dm["Esp"] = dm["Esp"].astype(int)

# --- m² FIJOS (inamovibles; no vienen del archivo)
M2_MAP = {1:120, 2:72, 3:72, 4:72, 5:180, 6:130, 7:60}
dm["m2"] = dm["Esp"].map(M2_MAP)
dm = dm[dm["m2"].notna()]  # solo Esps definidos en el mapa

# --- Canon por m² por registro
dm["Canon_m2_mes"] = dm["Monto"] / dm["m2"]
dm["Canon_m2_dia"] = dm["Canon_m2_mes"] / 30.0

# --- Agregado por Año x Esp (promedio observado)
agg = (
    dm.groupby(["Año","Esp"], as_index=False)[["Canon_m2_mes","Canon_m2_dia"]]
      .mean()
      .sort_values(["Año","Esp"])
      .copy()
)

# --- TODOS los años presentes en el archivo (aunque no haya canon mensual en algunos)
years_all = (
    pd.to_numeric(_df["Año"], errors="coerce")
      .dropna()
      .astype(int)
      .sort_values()
      .unique()
      .tolist()
)
if not years_all:
    years_all = sorted(agg["Año"].unique().tolist())

# --- Selección de métrica (mensual/diario)
valor_col = "Canon_m2_mes" if escala_m2 == "Mensual" else "Canon_m2_dia"
agg["valor_m2_clp"] = agg[valor_col]

# --- UF opcional (CSV con columnas Año,UF_promedio)
agg["UF_promedio"] = np.nan
if uf_url.strip():
    try:
        uf_df = pd.read_csv(uf_url)
        uf_df = uf_df.rename(columns={c: c.strip() for c in uf_df.columns})
        if {"Año","UF_promedio"}.issubset(uf_df.columns):
            uf_df["Año"] = pd.to_numeric(uf_df["Año"], errors="coerce").astype("Int64")
            uf_df["UF_promedio"] = pd.to_numeric(uf_df["UF_promedio"], errors="coerce")
            agg = agg.merge(
                uf_df.dropna(subset=["Año","UF_promedio"]).astype({"Año":"int"}),
                on="Año", how="left"
            )
        else:
            st.warning("El CSV de UF debe tener columnas: Año,UF_promedio. Se omite UF.")
    except Exception as e:
        st.warning(f"No se pudo leer UF desde la URL: {e}")

monedas = ["CLP"] + (["UF"] if agg["UF_promedio"].notna().any() else [])
moneda_m2 = st.radio("Moneda", monedas, horizontal=True, index=0, key="moneda_m2_final")

# --- Conversión a UF si procede
if moneda_m2 == "UF":
    agg["valor_m2"] = agg["valor_m2_clp"] / agg["UF_promedio"]
else:
    agg["valor_m2"] = agg["valor_m2_clp"]

# --- Selector dinámico de Esp
todos_esps = sorted(agg["Esp"].unique().tolist())
sel_esps = st.multiselect("Selecciona Espacios", todos_esps, default=todos_esps, key="sel_esps_m2_final")
agg = agg[agg["Esp"].isin(sel_esps)]

# --- Pivot reindexando con todos los años (años sin datos = 0)
plot_df = (
    agg.pivot_table(index="Año", columns="Esp", values="valor_m2", aggfunc="mean")
       .reindex(years_all)
       .fillna(0)
)

# --- Si quedó vacío, avisar y no romper
if plot_df.empty:
    st.info("No hay datos para el filtro actual de 'canon mensual' o para los Esp seleccionados.")
else:
    # --- Figura (eje X NUMÉRICO: 1 tick por año)
    fig = go.Figure()
    x_years = plot_df.index.astype(int).values  # años numéricos

    palette = [
        "#2563EB","#10B981","#F59E0B","#EF4444","#8B5CF6",
        "#14B8A6","#F97316","#DC2626","#3B82F6","#22C55E",
        "#EAB308","#EC4899","#0EA5E9","#A3E635"
    ]

    for i, esp in enumerate(plot_df.columns):
        y_series = plot_df[esp].values
        custom = np.where(y_series == 0, "⚠️ Sin registros de ‘canon mensual’", " ")
        fig.add_trace(go.Scatter(
            x=x_years, y=y_series, customdata=custom,
            mode="lines+markers",
            name=f"Esp {esp}",
            line=dict(width=3, color=palette[i % len(palette)]),
            marker=dict(size=7),
            hovertemplate="<b>Año %{x}</b><br>Esp: "+str(esp)+
                          "<br>Valor: %{y:,.2f}"+(" UF/m²" if moneda_m2=="UF" else " CLP/m²")+
                          "<br>%{customdata}<extra></extra>"
        ))

        # Anotación en el último punto de cada serie (si existe)
        if len(y_series) and pd.notna(y_series[-1]):
            fig.add_annotation(
                x=int(x_years[-1]), y=y_series[-1],
                text=f"Esp {esp}<br><b>{y_series[-1]:,.2f}{' UF/m²' if moneda_m2=='UF' else ' CLP/m²'}</b>",
                showarrow=True, arrowhead=2, ax=30, ay=-30,
                bgcolor="rgba(0,0,0,0.72)", bordercolor="rgba(0,0,0,0.72)",
                font=dict(color="white", size=11)
            )

    # Layout
    titulo_y = "UF/m²" if moneda_m2 == "UF" else "CLP/m²"
    titulo_esc = "mensual" if escala_m2 == "Mensual" else "diario"
    fig.update_layout(
        title=dict(text=f"Canon por m² ({titulo_esc}) — {moneda_m2} · por Año y Esp", x=0.02, xanchor="left"),
        xaxis_title="Año", yaxis_title=titulo_y, template="plotly_white", hovermode="x",
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", y=1.02, x=0.02),
        updatemenus=[dict(type="buttons", direction="right", x=1, xanchor="right", y=1.15, yanchor="top",
                          buttons=[dict(label="Mostrar todo", method="update", args=[{"visible":[True]*len(plot_df.columns)}]),
                                   dict(label="Ocultar todo", method="update", args=[{"visible":[False]*len(plot_df.columns)}])])]
    )

    # Eje X numérico con un tick por año
    if len(x_years):
        fig.update_xaxes(
            type="linear",
            tickmode="linear",
            tick0=int(x_years.min()),
            dtick=1,
            range=[int(x_years.min()) - 0.5, int(x_years.max()) + 0.5],
            showgrid=False
        )

    # Ajuste automático del rango Y ignorando ceros
    y_min = 0
    y_max = plot_df.replace(0, np.nan).max().max()
    y_max = float(y_max) * 1.1 if pd.notna(y_max) else 1.0
    fig.update_yaxes(range=[y_min, y_max], showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)

    st.plotly_chart(
        fig, use_container_width=True,
        config={"displaylogo": False, "displayModeBar": True,
                "modeBarButtonsToAdd": ["toImage","drawline","drawrect","eraseshape"]}
    )

   # --- Tabla (DISPLAY con $ y puntos de miles) + descarga Excel (NUMÉRICO) ---


st.markdown("#### 📄 Dataset agregado (Canon/m² — Año x Esp)")

df_x = plot_df.reset_index().rename(columns={"index": "Año"}).copy()

# --- Copia para DISPLAY (formateada como texto) ---
df_display = df_x.copy()

def miles_punto(n):
    # 1) formato estándar en inglés 1,234,567
    s = f"{n:,.0f}"
    # 2) convertir a formato chileno 1.234.567
    return s.replace(",", ".")

def uf_chileno(n):
    # miles con punto y 2 decimales con coma: 1.234,56
    s = f"{n:,.2f}"             # 1,234,56 (en-US)
    s = s.replace(",", "§")     # 1§234§56
    s = s.replace(".", ",")     # 1§234,56
    s = s.replace("§", ".")     # 1.234,56
    return s

# Formateo SOLO para mostrar (sin afectar Excel)
if moneda_m2 == "CLP":
    for c in df_display.columns[1:]:
        df_display[c] = df_display[c].round(0).apply(lambda v: f"${miles_punto(v)}")
else:  # UF
    for c in df_display.columns[1:]:
        df_display[c] = df_display[c].round(2).apply(uf_chileno)

# Encabezados bonitos
if moneda_m2 == "CLP":
    df_display = df_display.rename(columns={c: f"Esp {c} (CLP/m²)" for c in df_display.columns if c != "Año"})
else:
    df_display = df_display.rename(columns={c: f"Esp {c} (UF/m²)" for c in df_display.columns if c != "Año"})

st.dataframe(df_display, hide_index=True, use_container_width=True)

# --- Descarga en Excel (NUMÉRICA, con formato en el archivo) ---
from io import BytesIO
excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    df_x.to_excel(writer, index=False, sheet_name="canon_m2")
    wb = writer.book
    ws = writer.sheets["canon_m2"]

    # Encabezados
    head_fmt = wb.add_format({"bold": True})
    ws.set_row(0, None, head_fmt)

    # Anchos y formatos numéricos
    ws.set_column(0, 0, 10)  # Año
    if moneda_m2 == "CLP":
        numfmt = wb.add_format({"num_format": "$#,##0"})     # Excel con miles
        # renombrar encabezados en Excel también
        for j, c in enumerate(df_x.columns[1:], start=1):
            ws.write(0, j, f"Esp {c} (CLP/m²)", head_fmt)
    else:
        numfmt = wb.add_format({"num_format": '#,##0.00'})   # Excel con 2 decimales
        for j, c in enumerate(df_x.columns[1:], start=1):
            ws.write(0, j, f"Esp {c} (UF/m²)", head_fmt)

    ws.set_column(1, len(df_x.columns)-1, 14, numfmt)

excel_buffer.seek(0)
st.download_button(
    "⬇️ Descargar Excel (Canon/m² — Año x Esp)",
    data=excel_buffer.getvalue(),
    file_name=f"canon_m2_{titulo_esc}_{moneda_m2}_por_anio_y_esp.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)