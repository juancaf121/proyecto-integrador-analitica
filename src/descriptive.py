"""
Módulo de Analítica Descriptiva (Fase 2B) para la Plataforma Inteligente de
Planeación de la Producción y Distribución.

Calcula tablas estadísticas y genera gráficos interactivos en Plotly para:
1. Comportamiento histórico de la demanda (serie mensual, MA y tendencia OLS).
2. Productos con mayor y menor demanda (ranking y diagrama de Pareto).
3. Participación de clientes y plantas (Donut, Treemap, barras apiladas).
4. Tendencias y estacionalidad (heatmaps, índice estacional, descomposición clásica).
5. Variabilidad de la demanda (boxplots, CV%, matriz ABC-XYZ, tabla estadística).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

from src.etl import CLIENTES_ORDENADOS, MAPA_PRODUCTO_PLANTA, PLANTAS_OFICIALES, PRODUCTOS_ORDENADOS

MESES_ES_CORTO: dict[int, str] = {
    1: "Ene",
    2: "Feb",
    3: "Mar",
    4: "Abr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Ago",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dic",
}

COLORES_PLANTAS: dict[str, str] = {
    "Bogotá": "#1f77b4",
    "Medellín": "#2ca02c",
    "Cali": "#ff7f0e",
    "Bucaramanga": "#9467bd",
}

COLORES_CLIENTES: dict[str, str] = {
    "Cliente 1": "#0f4c81",
    "Cliente 2": "#1b7895",
    "Cliente 3": "#3aa6b9",
    "Cliente 4": "#f4a261",
    "Cliente 5": "#e76f51",
}

PALETA_CATEGORICA: list[str] = [
    "#0f4c81",
    "#1b7895",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#d62728",
    "#17becf",
    "#e377c2",
    "#8c564b",
    "#bcbd22",
]


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara una copia segura con tipos correctos e Ingreso calculado."""
    out = df.copy()
    out["Fecha"] = pd.to_datetime(out["Fecha"])
    out["Demanda"] = pd.to_numeric(out["Demanda"], errors="coerce").fillna(0.0)
    out["Precio"] = pd.to_numeric(out["Precio"], errors="coerce").fillna(0.0)
    out["Ingreso"] = out["Demanda"] * out["Precio"]
    return out


def _ordenar_productos_presentes(productos: pd.Series | list[str]) -> list[str]:
    """Retorna la lista de productos ordenada numéricamente según P1..P10."""
    presentes = set(productos)
    return [p for p in PRODUCTOS_ORDENADOS if p in presentes]


def _ordenar_clientes_presentes(clientes: pd.Series | list[str]) -> list[str]:
    """Retorna la lista de clientes ordenada numéricamente según Cliente 1..5."""
    presentes = set(clientes)
    return [c for c in CLIENTES_ORDENADOS if c in presentes]


def _apply_base_layout(fig: go.Figure, title: str, height: int = 450) -> go.Figure:
    """Aplica estilo visual ejecutivo consistente a todas las figuras Plotly."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=height,
        margin=dict(l=50, r=30, t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Segoe UI, Roboto, sans-serif", size=12),
    )
    return fig


def _clasificar_abc(acum_pct: float) -> str:
    if acum_pct <= 80.0:
        return "A"
    if acum_pct <= 95.0:
        return "B"
    return "C"


def _clasificar_xyz(cv_pct: float) -> str:
    if cv_pct < 20.0:
        return "X (Estable)"
    if cv_pct <= 35.0:
        return "Y (Moderada)"
    return "Z (Volátil)"


# ==============================================================================
# 1. Comportamiento histórico de la demanda
# ==============================================================================
def plot_evolucion_historica(
    df: pd.DataFrame,
    agregacion: str = "Total",
    ventana_ma: int = 3,
    mostrar_tendencia: bool = True,
) -> go.Figure:
    """
    Genera la serie de tiempo mensual agregada o desagregada por Producto,
    Cliente o Planta, incluyendo promedio móvil y tendencia lineal (OLS).
    """
    data = _prepare_df(df)
    if data.empty:
        return _apply_base_layout(go.Figure(), "Sin datos para los filtros seleccionados")

    fig = go.Figure()

    if agregacion == "Total":
        ts = data.groupby("Fecha", as_index=False)["Demanda"].sum().sort_values("Fecha")
        ts["MA"] = ts["Demanda"].rolling(window=ventana_ma, min_periods=1).mean()
        ts["STD"] = ts["Demanda"].rolling(window=ventana_ma, min_periods=1).std().fillna(0.0)
        ts["Upper"] = ts["MA"] + ts["STD"]
        ts["Lower"] = (ts["MA"] - ts["STD"]).clip(lower=0)

        fig.add_trace(
            go.Scatter(
                x=ts["Fecha"],
                y=ts["Upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ts["Fecha"],
                y=ts["Lower"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(27, 120, 149, 0.14)",
                name=f"Banda ±1σ (ventana {ventana_ma}m)",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ts["Fecha"],
                y=ts["Demanda"],
                mode="lines+markers",
                name="Demanda Mensual",
                line=dict(color="#0f4c81", width=2.5),
                marker=dict(size=6),
                hovertemplate="<b>%{x|%Y-%m}</b><br>Demanda: %{y:,.0f} uds<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ts["Fecha"],
                y=ts["MA"],
                mode="lines",
                name=f"Promedio Móvil ({ventana_ma}m)",
                line=dict(color="#ff7f0e", width=2.2, dash="dash"),
                hovertemplate="<b>%{x|%Y-%m}</b><br>MA: %{y:,.1f} uds<extra></extra>",
            )
        )

        if mostrar_tendencia and len(ts) >= 2:
            x_num = np.arange(len(ts), dtype=float)
            slope, intercept = np.polyfit(x_num, ts["Demanda"].values, 1)
            trend_vals = intercept + slope * x_num
            signo = "+" if slope >= 0 else ""
            fig.add_trace(
                go.Scatter(
                    x=ts["Fecha"],
                    y=trend_vals,
                    mode="lines",
                    name=f"Tendencia OLS ({signo}{slope:,.1f} uds/mes)",
                    line=dict(color="#d62728", width=2, dash="dot"),
                    hovertemplate="Tendencia: %{y:,.1f} uds<extra></extra>",
                )
            )
        titulo = "Evolución Histórica de la Demanda Total Mensual"
    else:
        col_grp = agregacion if agregacion in ("Producto", "Cliente", "Planta") else "Producto"
        ts = (
            data.groupby(["Fecha", col_grp], as_index=False)["Demanda"]
            .sum()
            .sort_values(["Fecha"])
        )
        if col_grp == "Producto":
            categorias = _ordenar_productos_presentes(ts[col_grp].unique())
        elif col_grp == "Cliente":
            categorias = _ordenar_clientes_presentes(ts[col_grp].unique())
        else:
            categorias = [p for p in PLANTAS_OFICIALES if p in set(ts[col_grp].unique())]

        for i, cat in enumerate(categorias):
            sub = ts[ts[col_grp] == cat].sort_values("Fecha")
            color = (
                COLORES_PLANTAS.get(cat)
                if col_grp == "Planta"
                else COLORES_CLIENTES.get(cat)
                if col_grp == "Cliente"
                else PALETA_CATEGORICA[i % len(PALETA_CATEGORICA)]
            )
            fig.add_trace(
                go.Scatter(
                    x=sub["Fecha"],
                    y=sub["Demanda"],
                    mode="lines+markers",
                    name=cat,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate=f"<b>{cat}</b> (%{{x|%Y-%m}})<br>Demanda: %{{y:,.0f}} uds<extra></extra>",
                )
            )
            if ventana_ma > 1 and len(sub) >= ventana_ma:
                ma_sub = sub["Demanda"].rolling(window=ventana_ma, min_periods=1).mean()
                fig.add_trace(
                    go.Scatter(
                        x=sub["Fecha"],
                        y=ma_sub,
                        mode="lines",
                        name=f"{cat} (MA {ventana_ma}m)",
                        line=dict(color=color, width=1.2, dash="dot"),
                        visible="legendonly",
                        hoverinfo="skip",
                    )
                )
        titulo = f"Evolución Histórica de la Demanda por {col_grp}"

    fig.update_xaxes(title_text="Período (Mes)", tickformat="%b %Y")
    fig.update_yaxes(title_text="Unidades Demandadas", tickformat=",.0f")
    fig.update_layout(hovermode="x unified")
    return _apply_base_layout(fig, titulo, height=470)


# ==============================================================================
# 2. Productos con mayor y menor demanda + Diagrama de Pareto
# ==============================================================================
def get_tabla_productos(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna ranking de productos ordenado de mayor a menor demanda con Pareto."""
    data = _prepare_df(df)
    if data.empty:
        return pd.DataFrame()

    mensual = data.groupby(["Producto", "Fecha"], as_index=False)["Demanda"].sum()
    prom_mensual = mensual.groupby("Producto")["Demanda"].mean()

    resumen = (
        data.groupby("Producto", as_index=False)
        .agg(
            Demanda_Total=("Demanda", "sum"),
            Precio_Mediano=("Precio", "median"),
            Ingreso_Estimado=("Ingreso", "sum"),
        )
    )
    resumen["Planta"] = resumen["Producto"].map(MAPA_PRODUCTO_PLANTA)
    resumen["Promedio_Mensual"] = resumen["Producto"].map(prom_mensual).round(1)
    resumen = resumen.sort_values("Demanda_Total", ascending=False).reset_index(drop=True)

    total = resumen["Demanda_Total"].sum()
    resumen["Participacion_pct"] = (resumen["Demanda_Total"] / total * 100).round(2) if total > 0 else 0.0
    resumen["Acumulado_pct"] = (resumen["Demanda_Total"].cumsum() / total * 100).round(2) if total > 0 else 0.0
    resumen["Clase_ABC"] = resumen["Acumulado_pct"].apply(_clasificar_abc)
    resumen["Demanda_Total"] = resumen["Demanda_Total"].astype(int)
    resumen["Ingreso_Estimado"] = resumen["Ingreso_Estimado"].round(0)

    cols = [
        "Producto",
        "Planta",
        "Demanda_Total",
        "Promedio_Mensual",
        "Participacion_pct",
        "Acumulado_pct",
        "Clase_ABC",
        "Precio_Mediano",
        "Ingreso_Estimado",
    ]
    return resumen[cols]


def plot_ranking_productos_barras(df: pd.DataFrame) -> go.Figure:
    """Gráfico de barras horizontales ordenado mostrando liderazgo y menor demanda."""
    tabla = get_tabla_productos(df)
    if tabla.empty:
        return _apply_base_layout(go.Figure(), "Sin datos de productos")

    tabla_plot = tabla.sort_values("Demanda_Total", ascending=True)
    fig = px.bar(
        tabla_plot,
        x="Demanda_Total",
        y="Producto",
        color="Planta",
        orientation="h",
        color_discrete_map=COLORES_PLANTAS,
        text=tabla_plot.apply(
            lambda r: f"{int(r['Demanda_Total']):,} uds ({r['Participacion_pct']:.2f}%)", axis=1
        ),
        hover_data={
            "Planta": True,
            "Demanda_Total": ":,.0f",
            "Participacion_pct": ":.2f",
            "Ingreso_Estimado": ":$,.0f",
        },
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(title_text="Unidades Totales", tickformat=",.0f", range=[0, tabla_plot["Demanda_Total"].max() * 1.24])
    fig.update_yaxes(title_text="Producto", categoryorder="array", categoryarray=tabla_plot["Producto"].tolist())
    return _apply_base_layout(fig, "Ranking de Productos: Mayor a Menor Demanda Total", height=460)


def plot_pareto_productos(df: pd.DataFrame) -> go.Figure:
    """Diagrama de Pareto (Volumen total y % acumulado) por producto."""
    tabla = get_tabla_productos(df)
    if tabla.empty:
        return _apply_base_layout(go.Figure(), "Sin datos para Pareto")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=tabla["Producto"],
            y=tabla["Demanda_Total"],
            name="Demanda Total (uds)",
            marker_color=[COLORES_PLANTAS.get(p, "#0f4c81") for p in tabla["Planta"]],
            text=tabla["Demanda_Total"].map(lambda v: f"{v:,.0f}"),
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Demanda: %{y:,.0f} uds<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=tabla["Producto"],
            y=tabla["Acumulado_pct"],
            mode="lines+markers+text",
            name="% Acumulado",
            line=dict(color="#d62728", width=2.5),
            marker=dict(size=7),
            text=tabla["Acumulado_pct"].map(lambda v: f"{v:.1f}%"),
            textposition="top center",
            hovertemplate="<b>%{x}</b><br>Acumulado: %{y:.2f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="#555555",
        secondary_y=True,
        annotation_text="Corte 80% (Pareto)",
        annotation_position="bottom right",
    )

    fig.update_xaxes(title_text="Producto (Orden Descendente)")
    fig.update_yaxes(
        title_text="Demanda Total (Unidades)",
        tickformat=",.0f",
        range=[0, tabla["Demanda_Total"].max() * 1.18],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Participación Acumulada (%)",
        range=[0, 110],
        ticksuffix="%",
        secondary_y=True,
    )
    return _apply_base_layout(fig, "Diagrama de Pareto de Demanda por Producto", height=460)


# ==============================================================================
# 3. Clientes y Plantas con mayor participación
# ==============================================================================
def get_tabla_clientes(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen de participación por Cliente ordenada descendentemente."""
    data = _prepare_df(df)
    if data.empty:
        return pd.DataFrame()

    mensual = data.groupby(["Cliente", "Fecha"], as_index=False)["Demanda"].sum()
    prom_mensual = mensual.groupby("Cliente")["Demanda"].mean()

    resumen = (
        data.groupby("Cliente", as_index=False)
        .agg(
            Demanda_Total=("Demanda", "sum"),
            Ingreso_Estimado=("Ingreso", "sum"),
        )
    )
    resumen["Promedio_Mensual"] = resumen["Cliente"].map(prom_mensual).round(1)
    resumen = resumen.sort_values("Demanda_Total", ascending=False).reset_index(drop=True)

    total = resumen["Demanda_Total"].sum()
    resumen["Participacion_pct"] = (resumen["Demanda_Total"] / total * 100).round(2) if total > 0 else 0.0
    resumen["Acumulado_pct"] = (resumen["Demanda_Total"].cumsum() / total * 100).round(2) if total > 0 else 0.0
    resumen["Demanda_Total"] = resumen["Demanda_Total"].astype("int64")
    resumen["Ingreso_Estimado"] = resumen["Ingreso_Estimado"].round(0)
    return resumen[["Cliente", "Demanda_Total", "Promedio_Mensual", "Participacion_pct", "Acumulado_pct", "Ingreso_Estimado"]]


def get_tabla_plantas(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla resumen de participación y productos asignados por Planta."""
    data = _prepare_df(df)
    if data.empty:
        return pd.DataFrame()

    prods_por_planta = (
        data.groupby("Planta")["Producto"]
        .apply(lambda s: ", ".join(_ordenar_productos_presentes(s.unique())))
    )
    resumen = (
        data.groupby("Planta", as_index=False)
        .agg(
            Demanda_Total=("Demanda", "sum"),
            Ingreso_Estimado=("Ingreso", "sum"),
        )
    )
    resumen["Productos_Asignados"] = resumen["Planta"].map(prods_por_planta)
    resumen = resumen.sort_values("Demanda_Total", ascending=False).reset_index(drop=True)

    total = resumen["Demanda_Total"].sum()
    resumen["Participacion_pct"] = (resumen["Demanda_Total"] / total * 100).round(2) if total > 0 else 0.0
    resumen["Demanda_Total"] = resumen["Demanda_Total"].astype(int)
    resumen["Ingreso_Estimado"] = resumen["Ingreso_Estimado"].round(0)
    return resumen[["Planta", "Productos_Asignados", "Demanda_Total", "Participacion_pct", "Ingreso_Estimado"]]


def plot_participacion_clientes_donut(df: pd.DataFrame) -> go.Figure:
    """Gráfico Donut de participación porcentual por Cliente."""
    tabla = get_tabla_clientes(df)
    if tabla.empty:
        return _apply_base_layout(go.Figure(), "Sin datos de clientes")

    total_uds = int(tabla["Demanda_Total"].sum())
    fig = go.Figure(
        data=[
            go.Pie(
                labels=tabla["Cliente"],
                values=tabla["Demanda_Total"],
                hole=0.52,
                marker=dict(colors=[COLORES_CLIENTES.get(c, "#0f4c81") for c in tabla["Cliente"]]),
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="<b>%{label}</b><br>Unidades: %{value:,.0f}<br>Participación: %{percent}<extra></extra>",
            )
        ]
    )
    fig.add_annotation(
        text=f"<b>Total</b><br>{total_uds:,}<br>uds",
        x=0.5,
        y=0.5,
        font_size=14,
        showarrow=False,
    )
    return _apply_base_layout(fig, "Participación de Demanda por Cliente", height=440)


def plot_treemap_planta_producto_cliente(df: pd.DataFrame) -> go.Figure:
    """Treemap jerárquico Planta -> Producto -> Cliente ponderado por Demanda."""
    data = _prepare_df(df)
    if data.empty:
        return _apply_base_layout(go.Figure(), "Sin datos para Treemap")

    agg = (
        data.groupby(["Planta", "Producto", "Cliente"], as_index=False)
        .agg(Demanda=("Demanda", "sum"), Ingreso=("Ingreso", "sum"))
    )
    fig = px.treemap(
        agg,
        path=[px.Constant("Red Logística"), "Planta", "Producto", "Cliente"],
        values="Demanda",
        color="Planta",
        color_discrete_map=COLORES_PLANTAS,
        hover_data={"Demanda": ":,.0f", "Ingreso": ":$,.0f"},
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:,.0f} uds<br>%{percentRoot:.1%}",
        hovertemplate="<b>%{label}</b><br>Demanda: %{value:,.0f} uds<br>% del Total: %{percentRoot:.2%}<extra></extra>",
    )
    return _apply_base_layout(fig, "Treemap Jerárquico: Planta → Producto → Cliente", height=480)


def plot_barras_apiladas_producto_cliente(df: pd.DataFrame) -> go.Figure:
    """Barras apiladas de volumen por Producto desglosado por Cliente."""
    data = _prepare_df(df)
    if data.empty:
        return _apply_base_layout(go.Figure(), "Sin datos para barras apiladas")

    agg = data.groupby(["Producto", "Cliente"], as_index=False)["Demanda"].sum()
    prods_ord = _ordenar_productos_presentes(agg["Producto"].unique())
    clis_ord = _ordenar_clientes_presentes(agg["Cliente"].unique())

    fig = px.bar(
        agg,
        x="Producto",
        y="Demanda",
        color="Cliente",
        barmode="stack",
        category_orders={"Producto": prods_ord, "Cliente": clis_ord},
        color_discrete_map=COLORES_CLIENTES,
        hover_data={"Demanda": ":,.0f"},
    )
    fig.update_xaxes(title_text="Producto")
    fig.update_yaxes(title_text="Unidades Demandadas", tickformat=",.0f")
    return _apply_base_layout(fig, "Composición de Demanda por Producto y Cliente", height=440)


# ==============================================================================
# 4. Tendencias y Estacionalidad
# ==============================================================================
def plot_heatmap_estacionalidad(df: pd.DataFrame, modo: str = "Producto vs Mes") -> go.Figure:
    """
    Heatmap de estacionalidad seleccionable:
    - 'Producto vs Mes': Promedio mensual por Producto y mes calendario (Ene-Dic).
    - 'Año vs Mes': Demanda total por Año y mes calendario.
    """
    data = _prepare_df(df)
    if data.empty:
        return _apply_base_layout(go.Figure(), "Sin datos para Heatmap")

    data["Anio"] = data["Fecha"].dt.year
    data["MesNum"] = data["Fecha"].dt.month
    data["Mes"] = data["MesNum"].map(MESES_ES_CORTO)
    meses_orden = [MESES_ES_CORTO[m] for m in range(1, 13) if m in set(data["MesNum"])]

    if modo == "Año vs Mes":
        piv = (
            data.groupby(["Anio", "MesNum", "Mes"], as_index=False)["Demanda"]
            .sum()
            .pivot(index="Anio", columns="Mes", values="Demanda")
        )
        piv = piv[[m for m in meses_orden if m in piv.columns]]
        piv.index = piv.index.astype(str)
        titulo = "Heatmap Estacional: Demanda Total Mensual (Año vs. Mes)"
        cbar_title = "Unidades"
    else:
        mensual_prod = data.groupby(["Producto", "Anio", "MesNum", "Mes"], as_index=False)["Demanda"].sum()
        piv = (
            mensual_prod.groupby(["Producto", "Mes"], as_index=False)["Demanda"]
            .mean()
            .pivot(index="Producto", columns="Mes", values="Demanda")
        )
        prods_ord = _ordenar_productos_presentes(piv.index)
        cols_ord = [m for m in meses_orden if m in piv.columns]
        piv = piv.loc[prods_ord, cols_ord].round(0)
        titulo = "Heatmap Estacional: Promedio Mensual por Producto vs. Mes del Año"
        cbar_title = "Prom. Uds"

    fig = px.imshow(
        piv,
        text_auto=",.0f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels=dict(x="Mes del Año", y="Producto" if modo != "Año vs Mes" else "Año", color=cbar_title),
    )
    fig.update_xaxes(side="bottom")
    return _apply_base_layout(fig, titulo, height=450)


def plot_indice_estacional(df: pd.DataFrame) -> go.Figure:
    """Calcula y grafica el índice estacional mensual promedio (base 100%)."""
    data = _prepare_df(df)
    if data.empty:
        return _apply_base_layout(go.Figure(), "Sin datos para índice estacional")

    ts = data.groupby("Fecha", as_index=False)["Demanda"].sum()
    ts["MesNum"] = ts["Fecha"].dt.month
    media_global = ts["Demanda"].mean()

    est = (
        ts.groupby("MesNum", as_index=False)["Demanda"]
        .mean()
        .sort_values("MesNum")
    )
    est["Mes"] = est["MesNum"].map(MESES_ES_CORTO)
    est["Indice_pct"] = (est["Demanda"] / media_global * 100).round(2) if media_global > 0 else 100.0
    est["Estado"] = np.where(est["Indice_pct"] >= 100.0, "Pico (≥ 100%)", "Valle (< 100%)")

    colores = ["#1b7895" if v >= 100.0 else "#f4a261" for v in est["Indice_pct"]]
    fig = go.Figure(
        data=[
            go.Bar(
                x=est["Mes"],
                y=est["Indice_pct"],
                marker_color=colores,
                text=est["Indice_pct"].map(lambda v: f"{v:.1f}%"),
                textposition="outside",
                customdata=est[["Demanda", "Estado"]],
                hovertemplate="<b>%{x}</b><br>Índice Estacional: %{y:.2f}%<br>Prom. Mensual: %{customdata[0]:,.0f} uds<br>%{customdata[1]}<extra></extra>",
            )
        ]
    )
    fig.add_hline(
        y=100.0,
        line_dash="dash",
        line_color="#333333",
        annotation_text="Promedio Histórico (100%)",
        annotation_position="top left",
    )

    mes_pico = est.loc[est["Indice_pct"].idxmax()]
    mes_valle = est.loc[est["Indice_pct"].idxmin()]
    fig.add_annotation(
        x=mes_pico["Mes"],
        y=mes_pico["Indice_pct"],
        text=f"Pico: {mes_pico['Mes']} ({mes_pico['Indice_pct']:.1f}%)",
        showarrow=True,
        arrowhead=2,
        ay=-32,
        font=dict(color="#0f4c81", size=11),
    )
    fig.add_annotation(
        x=mes_valle["Mes"],
        y=mes_valle["Indice_pct"],
        text=f"Valle: {mes_valle['Mes']} ({mes_valle['Indice_pct']:.1f}%)",
        showarrow=True,
        arrowhead=2,
        ay=-32,
        font=dict(color="#b34700", size=11),
    )

    fig.update_xaxes(title_text="Mes del Año")
    fig.update_yaxes(
        title_text="Índice Estacional (%)",
        range=[0, max(125, float(est["Indice_pct"].max()) * 1.18)],
        ticksuffix="%",
    )
    return _apply_base_layout(fig, "Índice Estacional Mensual Promedio (Base = 100%)", height=430)


def plot_descomposicion_serie(
    df: pd.DataFrame,
    producto: str = "P1",
    cliente: str = "Todos",
    modelo: str = "additive",
) -> tuple[go.Figure, dict]:
    """
    Descomposición clásica de series de tiempo (`seasonal_decompose`, period=12)
    para cualquier Producto o combinación Producto-Cliente.
    """
    data = _prepare_df(df)
    if producto != "Todos":
        data = data[data["Producto"] == producto]
    if cliente != "Todos":
        data = data[data["Cliente"] == cliente]

    ts = (
        data.groupby("Fecha")["Demanda"]
        .sum()
        .sort_index()
        .asfreq("MS")
        .dropna()
    )

    etiqueta = f"{producto} | {cliente}"
    if len(ts) < 24:
        fig_err = go.Figure()
        fig_err.add_annotation(
            text=f"Se requieren al menos 24 meses continuos para descomposición estacional (actuales: {len(ts)}).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return _apply_base_layout(fig_err, f"Descomposición Estacional: {etiqueta}", height=420), {}

    decomp = seasonal_decompose(ts, model=modelo, period=12)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Serie Observada",
            "Tendencia (Trend)",
            "Componente Estacional (Period = 12)",
            "Residuo (Irregular)",
        ),
    )

    componentes = [
        (1, decomp.observed, "Observada", "#0f4c81", "lines+markers"),
        (2, decomp.trend, "Tendencia", "#2ca02c", "lines"),
        (3, decomp.seasonal, "Estacionalidad", "#ff7f0e", "lines+markers"),
        (4, decomp.resid, "Residuo", "#9467bd", "markers"),
    ]
    for row_idx, serie, nombre, color, mode in componentes:
        fig.add_trace(
            go.Scatter(
                x=serie.index,
                y=serie.values,
                mode=mode,
                name=nombre,
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                showlegend=False,
            ),
            row=row_idx,
            col=1,
        )

    fig.add_hline(
        y=1.0 if modelo == "multiplicative" else 0.0,
        line_dash="dot",
        line_color="#888888",
        row=3,
        col=1,
    )
    fig.add_hline(
        y=1.0 if modelo == "multiplicative" else 0.0,
        line_dash="dot",
        line_color="#888888",
        row=4,
        col=1,
    )
    fig.update_xaxes(tickformat="%b %Y", row=4, col=1)

    est_vals = pd.Series(decomp.seasonal.values, index=decomp.seasonal.index)
    est_por_mes = est_vals.groupby(est_vals.index.month).mean()
    mes_pico = MESES_ES_CORTO[int(est_por_mes.idxmax())]
    mes_valle = MESES_ES_CORTO[int(est_por_mes.idxmin())]

    resumen = {
        "serie": etiqueta,
        "modelo": modelo,
        "periodos": int(len(ts)),
        "mes_pico_estacional": mes_pico,
        "mes_valle_estacional": mes_valle,
        "amplitud_estacional": round(float(est_por_mes.max() - est_por_mes.min()), 2),
    }
    return _apply_base_layout(fig, f"Descomposición Clásica ({modelo.title()}) — {etiqueta}", height=680), resumen


# ==============================================================================
# 5. Variabilidad de la demanda y Matriz ABC-XYZ
# ==============================================================================
def get_tabla_estadistica_series(
    df: pd.DataFrame,
    por: str = "Producto-Cliente",
) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas completas por Producto (10) o por serie
    Producto-Cliente (50): Media, Mediana, Desv. Est., Mín, Q1, Q3, Máx, IQR,
    CV% e Ingreso Estimado, además de clasificación ABC-XYZ.
    """
    data = _prepare_df(df)
    if data.empty:
        return pd.DataFrame()

    if por == "Producto":
        mensual = (
            data.groupby(["Producto", "Fecha"], as_index=False)
            .agg(Demanda=("Demanda", "sum"), Ingreso=("Ingreso", "sum"))
        )
        grp_cols = ["Producto"]
    else:
        mensual = data.copy()
        grp_cols = ["Producto", "Cliente"]

    stats_df = (
        mensual.groupby(grp_cols, as_index=False)
        .agg(
            N_Meses=("Demanda", "count"),
            Demanda_Total=("Demanda", "sum"),
            Media=("Demanda", "mean"),
            Mediana=("Demanda", "median"),
            Desv_Est=("Demanda", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            Min=("Demanda", "min"),
            Q1=("Demanda", lambda s: float(s.quantile(0.25))),
            Q3=("Demanda", lambda s: float(s.quantile(0.75))),
            Max=("Demanda", "max"),
            Ingreso_Estimado=("Ingreso", "sum"),
        )
    )

    stats_df["Planta"] = stats_df["Producto"].map(MAPA_PRODUCTO_PLANTA)
    stats_df["IQR"] = (stats_df["Q3"] - stats_df["Q1"]).round(2)
    stats_df["CV_pct"] = np.where(
        stats_df["Media"] > 0,
        (stats_df["Desv_Est"] / stats_df["Media"]) * 100.0,
        0.0,
    ).round(2)

    total_dem = stats_df["Demanda_Total"].sum()
    stats_df = stats_df.sort_values("Demanda_Total", ascending=False).reset_index(drop=True)
    stats_df["Participacion_pct"] = (
        (stats_df["Demanda_Total"] / total_dem * 100).round(2) if total_dem > 0 else 0.0
    )
    stats_df["Acumulado_pct"] = (
        (stats_df["Demanda_Total"].cumsum() / total_dem * 100).round(2) if total_dem > 0 else 0.0
    )
    stats_df["Clase_ABC"] = stats_df["Acumulado_pct"].apply(_clasificar_abc)
    stats_df["Clase_XYZ"] = stats_df["CV_pct"].apply(_clasificar_xyz)

    for col in ["Media", "Mediana", "Desv_Est", "Q1", "Q3"]:
        stats_df[col] = stats_df[col].round(2)
    for col in ["Demanda_Total", "Min", "Max", "Ingreso_Estimado"]:
        stats_df[col] = stats_df[col].round(0).astype("int64")

    stats_df["_prod_num"] = stats_df["Producto"].str.extract(r"(\d+)").astype(int)
    sort_cols = ["_prod_num"]
    if "Cliente" in stats_df.columns:
        stats_df["_cli_num"] = stats_df["Cliente"].str.extract(r"(\d+)").astype(int)
        sort_cols.append("_cli_num")

    stats_df = stats_df.sort_values(sort_cols).drop(columns=sort_cols).reset_index(drop=True)

    ordered_cols = grp_cols + [
        "Planta",
        "N_Meses",
        "Demanda_Total",
        "Participacion_pct",
        "Media",
        "Mediana",
        "Desv_Est",
        "Min",
        "Q1",
        "Q3",
        "Max",
        "IQR",
        "CV_pct",
        "Clase_ABC",
        "Clase_XYZ",
        "Ingreso_Estimado",
    ]
    return stats_df[ordered_cols]


def plot_boxplots_demanda(df: pd.DataFrame, dimension: str = "Producto") -> go.Figure:
    """Boxplots interactivos de distribución de demanda por Producto o por Cliente."""
    data = _prepare_df(df)
    if data.empty:
        return _apply_base_layout(go.Figure(), "Sin datos para Boxplot")

    if dimension == "Cliente":
        orden = _ordenar_clientes_presentes(data["Cliente"].unique())
        fig = px.box(
            data,
            x="Cliente",
            y="Demanda",
            color="Cliente",
            category_orders={"Cliente": orden},
            color_discrete_map=COLORES_CLIENTES,
            points="outliers",
        )
        titulo = "Distribución y Dispersión de Demanda por Cliente"
    else:
        orden = _ordenar_productos_presentes(data["Producto"].unique())
        fig = px.box(
            data,
            x="Producto",
            y="Demanda",
            color="Planta",
            category_orders={"Producto": orden, "Planta": PLANTAS_OFICIALES},
            color_discrete_map=COLORES_PLANTAS,
            points="outliers",
        )
        titulo = "Distribución y Dispersión de Demanda por Producto"

    fig.update_yaxes(title_text="Unidades Demandadas (por registro mensual)", tickformat=",.0f")
    fig.update_xaxes(title_text=dimension)
    return _apply_base_layout(fig, titulo, height=450)


def plot_matriz_abc_xyz(df: pd.DataFrame) -> go.Figure:
    """
    Gráfico de dispersión Demanda Media vs. Coeficiente de Variación (CV%)
    para las 50 series Producto-Cliente (Matriz ABC-XYZ de estabilidad).
    """
    stats_50 = get_tabla_estadistica_series(df, por="Producto-Cliente")
    if stats_50.empty:
        return _apply_base_layout(go.Figure(), "Sin datos para Matriz ABC-XYZ")

    stats_50["Serie"] = stats_50["Producto"] + " - " + stats_50["Cliente"]
    fig = px.scatter(
        stats_50,
        x="Media",
        y="CV_pct",
        size="Ingreso_Estimado",
        color="Planta",
        symbol="Clase_ABC",
        color_discrete_map=COLORES_PLANTAS,
        text="Serie",
        hover_name="Serie",
        hover_data={
            "Media": ":.1f",
            "Desv_Est": ":.1f",
            "CV_pct": ":.2f",
            "Clase_ABC": True,
            "Clase_XYZ": True,
            "Ingreso_Estimado": ":$,.0f",
        },
        size_max=28,
    )
    fig.update_traces(textposition="top center", textfont=dict(size=9))

    med_x = float(stats_50["Media"].median())
    fig.add_vline(
        x=med_x,
        line_dash="dash",
        line_color="#666666",
        annotation_text=f"Mediana Volumen ({med_x:.1f} uds/mes)",
        annotation_position="top right",
    )
    fig.add_hline(
        y=20.0,
        line_dash="dot",
        line_color="#2ca02c",
        annotation_text="CV = 20% (Límite X: Estable)",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=35.0,
        line_dash="dot",
        line_color="#d62728",
        annotation_text="CV = 35% (Límite Z: Volátil)",
        annotation_position="top left",
    )

    fig.update_xaxes(title_text="Demanda Media Mensual (Unidades)")
    fig.update_yaxes(title_text="Coeficiente de Variación CV (%)", ticksuffix="%")
    return _apply_base_layout(
        fig,
        "Matriz de Estabilidad ABC-XYZ: Demanda Media vs. Coeficiente de Variación (50 Series)",
        height=540,
    )


def plot_comparacion_etl_serie(
    df_pre_interp: pd.DataFrame,
    producto: str,
    cliente: str,
) -> go.Figure:
    """
    Comparador visual 'Antes vs. Después del ETL' para una serie Producto-Cliente.
    Panel superior: serie pre-ETL con outliers marcados y umbral 3.0 × IQR.
    Panel inferior: serie limpia post-ETL con puntos imputados/interpolados destacados.
    """
    sub = df_pre_interp[
        (df_pre_interp["Producto"] == producto) & (df_pre_interp["Cliente"] == cliente)
    ].sort_values("Fecha")

    if sub.empty:
        return _apply_base_layout(go.Figure(), f"Sin datos para {producto} - {cliente}")

    upper = float(sub["Limite_Superior"].iloc[0])
    lower = float(sub["Limite_Inferior"].iloc[0])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"ANTES del ETL — {producto} · {cliente} (Datos observados con anomalías y huecos)",
            f"DESPUÉS del ETL — {producto} · {cliente} (Serie continua e imputada por interpolación lineal)",
        ),
    )

    # Panel 1: Pre-ETL
    fig.add_trace(
        go.Scatter(
            x=sub["Fecha"],
            y=sub["Demanda_Observada"],
            mode="lines+markers",
            name="Serie Pre-ETL",
            line=dict(color="#6c757d", width=1.8),
            marker=dict(size=6, color="#6c757d"),
            connectgaps=False,
            hovertemplate="<b>%{x|%Y-%m}</b><br>Pre-ETL: %{y:,.0f} uds<extra></extra>",
        ),
        row=1,
        col=1,
    )

    out_rows = sub[sub["Es_Outlier"]]
    if not out_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=out_rows["Fecha"],
                y=out_rows["Demanda_Observada"],
                mode="markers+text",
                name="Outlier detectado (3.0×IQR)",
                marker=dict(size=12, color="#d62728", symbol="x"),
                text=out_rows["Demanda_Observada"].map(lambda v: f"{v:,.0f}"),
                textposition="top center",
                textfont=dict(color="#d62728", size=11),
                hovertemplate="<b>Outlier %{x|%Y-%m}</b><br>Original: %{y:,.0f} uds<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_hline(
        y=upper,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Límite Sup. 3×IQR ({upper:,.1f})",
        annotation_position="top left",
        row=1,
        col=1,
    )
    if lower > 0:
        fig.add_hline(
            y=lower,
            line_dash="dot",
            line_color="#e76f51",
            annotation_text=f"Límite Inf. ({lower:,.1f})",
            annotation_position="bottom left",
            row=1,
            col=1,
        )

    # Panel 2: Post-ETL
    fig.add_trace(
        go.Scatter(
            x=sub["Fecha"],
            y=sub["Demanda_Limpia"],
            mode="lines+markers",
            name="Serie Limpia Final",
            line=dict(color="#0f4c81", width=2.5),
            marker=dict(size=6, color="#0f4c81"),
            hovertemplate="<b>%{x|%Y-%m}</b><br>Limpia: %{y:,.0f} uds<extra></extra>",
        ),
        row=2,
        col=1,
    )

    imp_falt = sub[sub["Es_Celda_Faltante"] & (~sub["Es_Outlier"])]
    if not imp_falt.empty:
        fig.add_trace(
            go.Scatter(
                x=imp_falt["Fecha"],
                y=imp_falt["Demanda_Limpia"],
                mode="markers",
                name="Celda faltante/negativa imputada",
                marker=dict(size=10, color="#ff7f0e", symbol="diamond"),
                hovertemplate="<b>Imputado %{x|%Y-%m}</b><br>Valor: %{y:,.0f} uds<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if not out_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=out_rows["Fecha"],
                y=out_rows["Demanda_Limpia"],
                mode="markers",
                name="Outlier corregido (interpolado)",
                marker=dict(size=11, color="#2ca02c", symbol="star"),
                hovertemplate="<b>Outlier corregido %{x|%Y-%m}</b><br>Valor limpio: %{y:,.0f} uds<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Unidades (Crudo)", tickformat=",.0f", row=1, col=1)
    fig.update_yaxes(title_text="Unidades (Limpio)", tickformat=",.0f", row=2, col=1)
    fig.update_xaxes(title_text="Período (Mes)", tickformat="%b %Y", row=2, col=1)
    return _apply_base_layout(
        fig,
        f"Comparativa Antes vs. Después del Proceso ETL — {producto} ({cliente})",
        height=600,
    )
