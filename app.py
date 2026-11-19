"""
Plataforma Inteligente para la Planeación de la Producción y la Distribución
mediante Analítica Predictiva y Prescriptiva.

Entrega 1 y 2:
- Fase 1: Consolidación de Base de Datos Histórica (Panel 10 × 5 × 36 = 1.800)
- Fase 2: Proceso ETL Auditable y Analítica Descriptiva Interactiva
- Fase 3: Modelos de Pronóstico (Promedio Móvil, Holt, Holt-Winters, MLP)
- Fase 4: Evaluación Walk-Forward, Selección Automática y Matriz D_{p,c}
- Fase 5: Modelo de Optimización (PuLP) de Producción, Trasbordo y Distribución
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.descriptive import (
    get_tabla_clientes,
    get_tabla_estadistica_series,
    get_tabla_plantas,
    get_tabla_productos,
    plot_barras_apiladas_producto_cliente,
    plot_boxplots_demanda,
    plot_comparacion_etl_serie,
    plot_descomposicion_serie,
    plot_evolucion_historica,
    plot_heatmap_estacionalidad,
    plot_indice_estacional,
    plot_matriz_abc_xyz,
    plot_pareto_productos,
    plot_participacion_clientes_donut,
    plot_ranking_productos_barras,
    plot_treemap_planta_producto_cliente,
)
from src.etl import (
    CLIENTES_ORDENADOS,
    MAPA_PRODUCTO_PLANTA,
    PLANTAS_OFICIALES,
    PRODUCTOS_ORDENADOS,
    run_etl_pipeline,
)
from src.forecasting import (
    MODEL_NAMES,
    plot_demand_matrix_heatmap,
    plot_series_forecast_comparison,
    plot_winner_distribution_donut,
    run_forecasting_pipeline,
)
from src.optimization import (
    NODOS,
    SUPUESTOS_DEFAULT,
    CAP_NODO_DEFAULT,
    TABLA_SUPUESTOS,
    UBICACION_CLIENTES,
    UBICACION_NODOS,
    construir_parametros,
    plot_composicion_costos,
    plot_produccion_por_planta,
    plot_sankey_red,
    plot_utilizacion,
    resolver_modelo,
    tablas_parametros,
)
from src.scenarios import (
    correr_escenarios, restricciones_activas, tabla_resumen, tabla_plantas, tabla_nodos,
    tabla_clientes, plot_costo_escenarios, plot_produccion_planta_esc, plot_utilizacion_esc, plot_clientes_esc,
)

# ==============================================================================
# Configuración general de la página y estilos
# ==============================================================================
st.set_page_config(
    page_title="Plataforma Predictiva y Prescriptiva - Fases 1 a 6",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-banner {
        background: linear-gradient(135deg, #0f4c81 0%, #1b7895 100%);
        color: white;
        padding: 1.25rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .main-banner h1 {
        margin: 0;
        font-size: 1.55rem;
        font-weight: 700;
        color: #ffffff;
    }
    .main-banner p {
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
        color: #e6f2f8;
    }
    .section-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0f4c81;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.85rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RAW_DEFAULT_PATH = Path("demanda_historica_cruda.csv")
CLEAN_CSV_PATH = Path("data/demanda_limpia_1800.csv")
AUDIT_JSON_PATH = Path("data/auditoria_etl.json")
FORECASTS_CSV_PATH = Path("data/pronosticos_nov2026.csv")
METRICS_CSV_PATH = Path("data/metricas_modelos_50series.csv")


@st.cache_data(show_spinner="Ejecutando pipeline ETL auditable...")
def ejecutar_etl_oficial(file_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Ejecuta el ETL sobre el archivo oficial del proyecto y cachea por timestamp."""
    _ = file_mtime
    return run_etl_pipeline(
        raw_source=RAW_DEFAULT_PATH,
        output_csv_path=CLEAN_CSV_PATH,
        audit_json_path=AUDIT_JSON_PATH,
    )


@st.cache_data(show_spinner="Entrenando y validando los 4 modelos en las 50 series...")
def ejecutar_pronosticos_cacheados(
    df_clean: pd.DataFrame,
    ma_window: int,
    hw_seasonal: str,
    mlp_lags: int,
    mlp_max_iter: int,
    criterio: str,
    exportar_csv: bool,
) -> dict:
    """Entrena las 50 series con validación walk-forward y cachea por hiperparámetros."""
    return run_forecasting_pipeline(
        df_clean=df_clean,
        ma_window=ma_window,
        hw_seasonal=hw_seasonal,
        mlp_lags=mlp_lags,
        mlp_max_iter=mlp_max_iter,
        criterio=criterio,
        output_forecasts_path=FORECASTS_CSV_PATH if exportar_csv else None,
        output_metrics_path=METRICS_CSV_PATH if exportar_csv else None,
    )


def convertir_df_a_excel_bytes(df: pd.DataFrame, sheet_name: str = "Datos") -> bytes:
    """Genera un archivo Excel en memoria listo para descargar con openpyxl."""
    buffer = io.BytesIO()
    df_export = df.copy()
    if "Fecha" in df_export.columns and pd.api.types.is_datetime64_any_dtype(df_export["Fecha"]):
        df_export["Fecha"] = df_export["Fecha"].dt.strftime("%Y-%m-%d")
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def convertir_df_a_csv_bytes(df: pd.DataFrame) -> bytes:
    """Genera bytes UTF-8 CSV con fechas en formato YYYY-MM-DD."""
    df_export = df.copy()
    if "Fecha" in df_export.columns and pd.api.types.is_datetime64_any_dtype(df_export["Fecha"]):
        df_export["Fecha"] = df_export["Fecha"].dt.strftime("%Y-%m-%d")
    return df_export.to_csv(index=False).encode("utf-8")


def convertir_varios_df_a_excel_bytes(hojas: dict[str, pd.DataFrame]) -> bytes:
    """Genera un Excel en memoria con una hoja por DataFrame (nombres ≤ 31 caracteres)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for nombre, df in hojas.items():
            df.to_excel(writer, index=not isinstance(df.index, pd.RangeIndex), sheet_name=nombre[:31])
    return buffer.getvalue()


@st.cache_data(show_spinner="Resolviendo el modelo de optimización con PuLP (CBC)...")
def resolver_optimizacion_cacheada(
    df_forecasts: pd.DataFrame,
    df_clean: pd.DataFrame,
    supuestos_items: tuple,
    cap_nodo_items: tuple,
    factor_demanda: float,
    permitir_faltantes: bool,
) -> dict:
    """Construye parámetros y resuelve el modelo de Fase 5; se cachea por supuestos y demanda."""
    par = construir_parametros(
        df_forecasts=df_forecasts,
        df_clean=df_clean,
        supuestos=dict(supuestos_items),
        factor_demanda=factor_demanda,
        cap_nodo=dict(cap_nodo_items),
    )
    return resolver_modelo(par, permitir_faltantes=permitir_faltantes)


@st.cache_data(show_spinner="Resolviendo los 3 escenarios con PuLP (CBC)...")
def correr_escenarios_cacheado(df_forecasts, df_clean, supuestos_items, cap_nodo_items):
    return correr_escenarios(df_forecasts, df_clean, dict(supuestos_items), dict(cap_nodo_items))


# ==============================================================================
# Barra lateral: Carga de datos y estado de la arquitectura
# ==============================================================================
with st.sidebar:
    st.markdown("### 🏭 Control de Datos y ETL")
    fuente_opcion = st.radio(
        "Origen de datos crudos",
        options=[
            "Archivo oficial (demanda_historica_cruda.csv)",
            "Cargar otro archivo CSV / Excel",
        ],
        index=0,
    )

    uploaded_file = None
    if fuente_opcion == "Cargar otro archivo CSV / Excel":
        uploaded_file = st.file_uploader(
            "Subir archivo crudo",
            type=["csv", "xlsx", "xls"],
            help="El archivo debe contener las columnas: Fecha, Producto, Cliente, Demanda, Planta, Precio.",
        )

    if st.button("🔄 Recalcular ETL y Exportar CSV/JSON", width="stretch"):
        st.cache_data.clear()

    st.divider()
    st.markdown("### 🗺️ Red Logística del Proyecto")
    st.markdown(
        """
        - **Plantas (4):** Bogotá, Medellín, Cali, Bucaramanga
        - **Nodos Trasbordo (2):** Nodo Norte, Nodo Centro
        - **Productos (10):** P1 – P10
        - **Clientes (5):** Cliente 1 – Cliente 5
        - **Horizonte:** 36 meses (`2023-11` a `2026-10`)
        - **Panel Final:** `10 × 5 × 36 = 1.800`
        """
    )

# Carga y ejecución del ETL
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        df_uploaded = pd.read_excel(uploaded_file)
    else:
        df_uploaded = pd.read_csv(uploaded_file)
    df_clean, df_pre_interp, audit = run_etl_pipeline(
        raw_source=df_uploaded,
        output_csv_path=None,
        audit_json_path=None,
    )
    fuente_activa = f"Archivo cargado: {uploaded_file.name}"
else:
    if not RAW_DEFAULT_PATH.exists():
        st.error(f"No se encontró el archivo fuente `{RAW_DEFAULT_PATH}` en la raíz del proyecto.")
        st.stop()
    mtime = RAW_DEFAULT_PATH.stat().st_mtime
    df_clean, df_pre_interp, audit = ejecutar_etl_oficial(mtime)
    fuente_activa = f"Oficial: `{RAW_DEFAULT_PATH.name}` → Exportado en `{CLEAN_CSV_PATH.as_posix()}`"

with st.sidebar:
    st.divider()
    st.success(
        f"**Panel Balanceado Listo**\n\n"
        f"- Observaciones: **{audit['registros_limpios_finales']:,}**\n"
        f"- Series: **{audit['series_unicas']}** × **{audit['meses_continuos']}** meses\n"
        f"- Nulos restantes: **{sum(audit['nulos_finales'].values())}**"
    )
    st.caption(fuente_activa)


# ==============================================================================
# Encabezado principal
# ==============================================================================
st.markdown(
    """
    <div class="main-banner">
        <h1>Plataforma Inteligente para la Planeación de la Producción y la Distribución</h1>
        <p>Analítica Predictiva y Prescriptiva · Fases 1–2 (ETL y Descriptiva) , Fases 3–4 (Pronósticos, Validación Walk-Forward y Selección Automática) y Fase 5 (Optimización con PuLP)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📦 Fase 1: Base Consolidada (10×5×36)",
        "🛠️ Fase 2A: Auditoría del Proceso ETL",
        "📊 Fase 2B: Analítica Descriptiva",
        "🤖 Fase 3: Modelos de Pronóstico",
        "🏆 Fase 4: Evaluación y Matriz D_{p,c}",
        "🧮 Fase 5: Optimización (PuLP)",
        "🎯 Fase 6: Escenarios",
    ]
)

# ==============================================================================
# PESTAÑA 1: FASE 1 — BASE DE DATOS CONSOLIDADA
# ==============================================================================
with tab1:
    st.subheader("1. Arquitectura Dimensional del Panel Histórico")
    st.markdown(
        """
        <div class="section-card">
        <b>Diseño del panel balanceado:</b> La base histórica consolida la demanda mensual desagregada
        por <b>Producto (10)</b>, <b>Cliente (5)</b> y <b>Período mensual (36 meses continuos)</b>,
        generando exactamente <b>1.800 observaciones</b> (<code>2023-11-01</code> a <code>2026-10-01</code>).
        Esta desagregación garantiza que los modelos predictivos (Fase 3) alimenten directamente las decisiones
        de producción, trasbordo y distribución por cliente en el modelo de optimización matemática (Fase 5).
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Productos", f"{audit['productos_unicos']}", "P1 a P10")
    c2.metric("Clientes", f"{audit['clientes_unicos']}", "Cliente 1 a 5")
    c3.metric("Horizonte Temporal", f"{audit['meses_continuos']} meses", "Nov 2023 – Oct 2026")
    c4.metric("Series Independientes", f"{audit['series_unicas']}", "Pares Producto-Cliente")
    c5.metric("Panel Consolidado", f"{audit['registros_limpios_finales']:,}", "0% valores nulos")

    st.markdown("---")
    col_left, col_right = st.columns([1.25, 1.0])

    with col_left:
        st.subheader("2. Verificador de Completitud de las 50 Series (Matriz 10 × 5)")
        matriz_conteo = (
            df_clean.groupby(["Producto", "Cliente"])["Fecha"]
            .count()
            .unstack("Cliente")
            .loc[PRODUCTOS_ORDENADOS, CLIENTES_ORDENADOS]
        )
        todas_completas = bool((matriz_conteo == 36).all().all())
        if todas_completas:
            st.success(
                "✅ Panel 100% balanceado: las 50 combinaciones Producto-Cliente cuentan exactamente con "
                "36 períodos mensuales continuos sin huecos temporales."
            )
        else:
            st.warning("⚠️ Existen series con conteo distinto a 36 períodos.")

        fig_completitud = px.imshow(
            matriz_conteo,
            text_auto=True,
            color_continuous_scale=[[0.0, "#f4a261"], [1.0, "#1b7895"]],
            aspect="auto",
            labels=dict(x="Cliente", y="Producto", color="Meses"),
        )
        fig_completitud.update_layout(
            title="Matriz de Completitud (Meses Disponibles por Serie)",
            height=380,
            margin=dict(l=40, r=20, t=50, b=30),
        )
        st.plotly_chart(fig_completitud)

    with col_right:
        st.subheader("3. Asignación Histórica Planta – Producto")
        tabla_homologacion = (
            df_clean.groupby("Producto", as_index=False)
            .agg(
                Planta=("Planta", "first"),
                Precio_Mediano=("Precio", "median"),
                Demanda_Total=("Demanda", "sum"),
            )
        )
        tabla_homologacion["_num"] = tabla_homologacion["Producto"].str.extract(r"(\d+)").astype(int)
        tabla_homologacion = tabla_homologacion.sort_values("_num").drop(columns=["_num"]).reset_index(drop=True)
        st.dataframe(
            tabla_homologacion,
            hide_index=True,
            height=380,
            column_config={
                "Precio_Mediano": st.column_config.NumberColumn("Precio Mediano (COP)", format="$ %,.0f"),
                "Demanda_Total": st.column_config.NumberColumn("Demanda Total (uds)", format="%,d"),
            },
        )

    st.markdown("---")
    st.subheader("4. Explorador Interactivo y Exportación de la Base Limpia")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        f_plantas_t1 = st.multiselect("Filtrar Planta", PLANTAS_OFICIALES, default=PLANTAS_OFICIALES, key="t1_planta")
    with f_col2:
        prods_disponibles_t1 = [p for p in PRODUCTOS_ORDENADOS if MAPA_PRODUCTO_PLANTA[p] in f_plantas_t1]
        f_prods_t1 = st.multiselect("Filtrar Producto", prods_disponibles_t1, default=prods_disponibles_t1, key="t1_prod")
    with f_col3:
        f_clis_t1 = st.multiselect("Filtrar Cliente", CLIENTES_ORDENADOS, default=CLIENTES_ORDENADOS, key="t1_cli")
    with f_col4:
        fechas_unicas_t1 = sorted(df_clean["Fecha"].dt.strftime("%Y-%m-%d").unique())
        f_rango_t1 = st.select_slider(
            "Rango de Fechas",
            options=fechas_unicas_t1,
            value=(fechas_unicas_t1[0], fechas_unicas_t1[-1]),
            key="t1_fecha",
        )

    df_explorar = df_clean[
        (df_clean["Planta"].isin(f_plantas_t1))
        & (df_clean["Producto"].isin(f_prods_t1))
        & (df_clean["Cliente"].isin(f_clis_t1))
        & (df_clean["Fecha"] >= pd.to_datetime(f_rango_t1[0]))
        & (df_clean["Fecha"] <= pd.to_datetime(f_rango_t1[1]))
    ].copy()

    df_explorar_view = df_explorar.copy()
    df_explorar_view["Fecha"] = df_explorar_view["Fecha"].dt.strftime("%Y-%m-%d")

    st.caption(f"Mostrando **{len(df_explorar_view):,}** de **{len(df_clean):,}** registros consolidados.")
    st.dataframe(
        df_explorar_view,
        hide_index=True,
        height=340,
        column_config={
            "Demanda": st.column_config.NumberColumn("Demanda (uds)", format="%,d"),
            "Precio": st.column_config.NumberColumn("Precio (COP)", format="$ %,.0f"),
        },
    )

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            label="⬇️ Descargar Base Completa (`demanda_limpia_1800.csv`)",
            data=convertir_df_a_csv_bytes(df_clean),
            file_name="demanda_limpia_1800.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )
    with dl2:
        st.download_button(
            label="⬇️ Descargar Base Completa en Excel (.xlsx)",
            data=convertir_df_a_excel_bytes(df_clean),
            file_name="demanda_limpia_1800.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with dl3:
        st.download_button(
            label="⬇️ Descargar Vista Filtrada (CSV)",
            data=convertir_df_a_csv_bytes(df_explorar),
            file_name="demanda_limpia_filtrada.csv",
            mime="text/csv",
            width="stretch",
        )


# ==============================================================================
# PESTAÑA 2: FASE 2A — AUDITORÍA DEL PROCESO ETL
# ==============================================================================
with tab2:
    st.subheader("1. Resumen Ejecutivo de Auditoría del Pipeline ETL")
    st.markdown(
        """
        <div class="section-card">
        Trazabilidad determinista paso a paso desde los registros crudos (con ruido de formatos,
        duplicados, negativos, precios mal digitados, fechas imposibles, huecos y picos atípicos)
        hasta el panel limpio <b>10 × 5 × 36 = 1.800 observaciones</b> con <b>0% de valores nulos</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Registros Crudos", f"{audit['filas_crudas']:,}", f"{audit['filas_vacias']} filas vacías")
    k2.metric("Duplicados Exactos", f"{audit['duplicados_exactos']}", "Detectados en crudo")
    k3.metric("Llaves Repetidas Consolidadas", f"{audit['colisiones_llaves_duplicadas']}", f"→ {audit['combinaciones_unicas_validas']:,} únicas")
    k4.metric("Negativos / Ceros Corregidos", f"{audit['demanda_neg_cero_corregidos']}", "Convertidos a NaN e imputados")
    k5.metric("Outliers Interpolados (3×IQR)", f"{audit['outliers_detectados']}", "Por serie Producto-Cliente")

    k6, k7, k8, k9, k10 = st.columns(5)
    k6.metric("Celdas Reconstruidas (Panel)", f"{audit['celdas_reconstruidas_panel']}", "Reindexación 10×5×36")
    k7.metric("Productos Recuperados", f"{audit['productos_recuperados']}", "Inferidos por Planta y Precio")
    k8.metric("Precios Atípicos Corregidos", f"{audit['precios_atipicos_corregidos']}", "≤ 0 o > $100.000")
    k9.metric("Fechas Imposibles Corregidas", f"{audit['fechas_imposibles_corregidas']}", f"{audit['formatos_fecha_unicos']} formatos crudos")
    k10.metric("Registros Limpios Finales", f"{audit['registros_limpios_finales']:,}", "0% valores nulos")

    st.markdown("#### Estandarización de Dimensiones y Flujo de Depuración")
    e1, e2, e3, e4 = st.columns(4)
    e1.info(f"**Fecha:** `{audit['formatos_fecha_unicos']}` variantes → **36 meses** (`YYYY-MM-01`)")
    e2.info(f"**Producto:** `{audit['variantes_producto']}` variantes → **10 códigos** (`P1..P10`)")
    e3.info(f"**Cliente:** `{audit['variantes_cliente']}` variantes → **5 clientes** (`Cliente 1..5`)")
    e4.info(f"**Planta:** `{audit['variantes_planta']}` variantes → **4 plantas** oficiales")

    st.markdown("---")
    st.subheader("2. Comparador Visual: Antes vs. Después del Proceso ETL")

    series_con_outliers = sorted(
        {(o["Producto"], o["Cliente"]) for o in audit["outliers_detalle"]},
        key=lambda x: (int(x[0][1:]), int(x[1].split()[-1])),
    )

    comp_c1, comp_c2, comp_c3 = st.columns([1.2, 1.0, 1.0])
    with comp_c1:
        modo_selector = st.radio(
            "Filtro de selección de serie",
            options=["Series con Outliers Detectados (Recomendado)", "Todas las 50 series (Producto - Cliente)"],
            horizontal=False,
        )

    if modo_selector.startswith("Series") and series_con_outliers:
        opciones_str = [f"{p} - {c}" for p, c in series_con_outliers]
        with comp_c2:
            serie_elegida = st.selectbox(
                "Serie con anomalía",
                options=opciones_str,
                index=0,
                help="Incluye casos críticos como P2 - Cliente 3 (129.000) y P10 - Cliente 4 (168.000).",
            )
        prod_sel, cli_sel = serie_elegida.split(" - ")
        with comp_c3:
            out_info = [o for o in audit["outliers_detalle"] if o["Producto"] == prod_sel and o["Cliente"] == cli_sel]
            st.metric("Outliers en esta serie", len(out_info), f"Límite Sup: {out_info[0]['Limite_Superior']:,.1f}")
    else:
        with comp_c2:
            prod_sel = st.selectbox("Producto", PRODUCTOS_ORDENADOS, index=1, key="etl_prod_all")
        with comp_c3:
            cli_sel = st.selectbox("Cliente", CLIENTES_ORDENADOS, index=2, key="etl_cli_all")

    fig_comp = plot_comparacion_etl_serie(df_pre_interp, prod_sel, cli_sel)
    st.plotly_chart(fig_comp)

    st.markdown("---")
    st.subheader("3. Registros Detallados de Auditoría")
    sub_a, sub_b, sub_c, sub_d, sub_e = st.tabs(
        [
            f"🚨 Outliers ({audit['outliers_detectados']})",
            f"➖ Demanda ≤ 0 ({audit['demanda_neg_cero_corregidos']})",
            f"💲 Precios Atípicos ({audit['precios_atipicos_corregidos']})",
            f"🔎 Productos Recuperados ({audit['productos_recuperados']})",
            "📋 Nulos por Columna & Fechas",
        ]
    )

    with sub_a:
        df_out = pd.DataFrame(audit["outliers_detalle"])
        st.dataframe(
            df_out,
            hide_index=True,
            column_config={
                "Valor_Original": st.column_config.NumberColumn("Valor Original", format="%,d"),
                "Valor_Imputado": st.column_config.NumberColumn("Valor Imputado", format="%,d"),
                "Limite_Superior": st.column_config.NumberColumn("Límite Sup. (Q3 + 3×IQR)", format="%,.2f"),
            },
        )

    with sub_b:
        st.dataframe(pd.DataFrame(audit["demanda_neg_cero_detalle"]), hide_index=True)

    with sub_c:
        st.dataframe(
            pd.DataFrame(audit["precios_atipicos_detalle"]),
            hide_index=True,
            column_config={
                "Precio_Numerico": st.column_config.NumberColumn("Precio Detectado", format="$ %,.0f"),
            },
        )

    with sub_d:
        st.dataframe(
            pd.DataFrame(audit["productos_recuperados_detalle"]),
            hide_index=True,
            column_config={
                "Precio": st.column_config.NumberColumn("Precio Observado", format="$ %,.0f"),
                "Mediana_Referencia": st.column_config.NumberColumn("Mediana Producto", format="$ %,.0f"),
            },
        )

    with sub_e:
        nulos_comp = pd.DataFrame(
            {
                "Columna": list(audit["nulos_iniciales"].keys()),
                "Nulos_Iniciales": list(audit["nulos_iniciales"].values()),
                "Nulos_Finales": [audit["nulos_finales"].get(c, 0) for c in audit["nulos_iniciales"].keys()],
            }
        )
        col_n1, col_n2 = st.columns(2)
        with col_n1:
            st.markdown("**Conteo de Nulos: Crudo vs. Limpio**")
            st.dataframe(nulos_comp, hide_index=True)
        with col_n2:
            st.markdown("**Fechas de Calendario Inexistentes Corregidas**")
            st.dataframe(pd.DataFrame(audit["fechas_imposibles_detalle"]), hide_index=True)


# ==============================================================================
# PESTAÑA 3: FASE 2B — ANALÍTICA DESCRIPTIVA
# ==============================================================================
with tab3:
    st.subheader("Filtros Globales de Analítica Descriptiva")
    fg1, fg2, fg3, fg4 = st.columns([1.0, 1.2, 1.0, 1.2])
    with fg1:
        sel_plantas = st.multiselect(
            "Plantas",
            options=PLANTAS_OFICIALES,
            default=PLANTAS_OFICIALES,
            key="desc_plantas",
        )
    with fg2:
        prods_habilitados = [p for p in PRODUCTOS_ORDENADOS if MAPA_PRODUCTO_PLANTA[p] in sel_plantas]
        sel_productos = st.multiselect(
            "Productos",
            options=prods_habilitados,
            default=prods_habilitados,
            key="desc_productos",
        )
    with fg3:
        sel_clientes = st.multiselect(
            "Clientes",
            options=CLIENTES_ORDENADOS,
            default=CLIENTES_ORDENADOS,
            key="desc_clientes",
        )
    with fg4:
        fechas_str = sorted(df_clean["Fecha"].dt.strftime("%Y-%m-%d").unique())
        sel_fechas = st.select_slider(
            "Horizonte Temporal",
            options=fechas_str,
            value=(fechas_str[0], fechas_str[-1]),
            key="desc_fechas",
        )

    df_filt = df_clean[
        (df_clean["Planta"].isin(sel_plantas))
        & (df_clean["Producto"].isin(sel_productos))
        & (df_clean["Cliente"].isin(sel_clientes))
        & (df_clean["Fecha"] >= pd.to_datetime(sel_fechas[0]))
        & (df_clean["Fecha"] <= pd.to_datetime(sel_fechas[1]))
    ].copy()

    if df_filt.empty:
        st.warning("No hay datos disponibles para la combinación de filtros seleccionada. Ajusta los filtros.")
    else:
        df_filt_ing = df_filt.assign(Ingreso=df_filt["Demanda"] * df_filt["Precio"])
        dem_total = int(df_filt["Demanda"].sum())
        n_meses_filt = max(1, int(df_filt["Fecha"].nunique()))
        prom_mensual_filt = dem_total / n_meses_filt
        ingreso_total = float(df_filt_ing["Ingreso"].sum())

        tab_p_tmp = get_tabla_productos(df_filt)
        tab_c_tmp = get_tabla_clientes(df_filt)
        prod_lider = tab_p_tmp.iloc[0]
        cli_lider = tab_c_tmp.iloc[0]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Demanda Total", f"{dem_total:,} uds", f"{len(df_filt):,} registros")
        m2.metric("Promedio Mensual Agregado", f"{prom_mensual_filt:,.1f} uds", f"En {n_meses_filt} meses")
        m3.metric("Producto Líder", prod_lider["Producto"], f"{prod_lider['Demanda_Total']:,} uds ({prod_lider['Participacion_pct']}%)")
        m4.metric("Cliente Líder", cli_lider["Cliente"], f"{cli_lider['Demanda_Total']:,} uds ({cli_lider['Participacion_pct']}%)")
        m5.metric("Ingreso Estimado Total", f"${ingreso_total / 1e9:,.2f} B", f"${ingreso_total:,.0f} COP")

        st.markdown("---")
        # 1. Comportamiento histórico de la demanda
        st.markdown("### 1. Comportamiento Histórico de la Demanda")
        h1, h2, h3 = st.columns([1.2, 1.2, 1.0])
        with h1:
            nivel_agregacion = st.radio(
                "Nivel de desagregación",
                options=["Total", "Producto", "Cliente", "Planta"],
                horizontal=True,
                key="desc_agreg",
            )
        with h2:
            ventana_ma = st.slider("Ventana de Promedio Móvil (meses)", min_value=1, max_value=6, value=3, key="desc_ma")
        with h3:
            mostrar_ols = st.checkbox("Mostrar tendencia lineal (OLS)", value=True, key="desc_ols")

        st.plotly_chart(
            plot_evolucion_historica(
                df_filt,
                agregacion=nivel_agregacion,
                ventana_ma=ventana_ma,
                mostrar_tendencia=mostrar_ols,
            )
        )

        st.markdown("---")
        # 2. Productos con mayor y menor demanda
        st.markdown("### 2. Productos con Mayor y Menor Demanda (Ranking y Pareto 80/20)")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.plotly_chart(plot_ranking_productos_barras(df_filt))
        with p_col2:
            st.plotly_chart(plot_pareto_productos(df_filt))

        st.dataframe(
            tab_p_tmp,
            hide_index=True,
            column_config={
                "Demanda_Total": st.column_config.NumberColumn("Demanda Total", format="%,d"),
                "Promedio_Mensual": st.column_config.NumberColumn("Prom. Mensual", format="%,.1f"),
                "Participacion_pct": st.column_config.NumberColumn("Participación (%)", format="%.2f%%"),
                "Acumulado_pct": st.column_config.NumberColumn("% Acumulado", format="%.2f%%"),
                "Precio_Mediano": st.column_config.NumberColumn("Precio Mediano", format="$ %,.0f"),
                "Ingreso_Estimado": st.column_config.NumberColumn("Ingreso Estimado", format="$ %,.0f"),
            },
        )

        st.markdown("---")
        # 3. Clientes y plantas con mayor participación
        st.markdown("### 3. Participación de Clientes y Plantas")
        c_col1, c_col2 = st.columns([1.0, 1.2])
        with c_col1:
            st.plotly_chart(plot_participacion_clientes_donut(df_filt))
        with c_col2:
            st.plotly_chart(plot_barras_apiladas_producto_cliente(df_filt))

        st.plotly_chart(plot_treemap_planta_producto_cliente(df_filt))

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown("**Participación por Cliente**")
            st.dataframe(
                tab_c_tmp,
                hide_index=True,
                column_config={
                    "Demanda_Total": st.column_config.NumberColumn("Demanda Total", format="%,d"),
                    "Promedio_Mensual": st.column_config.NumberColumn("Prom. Mensual", format="%,.1f"),
                    "Participacion_pct": st.column_config.NumberColumn("Participación", format="%.2f%%"),
                    "Acumulado_pct": st.column_config.NumberColumn("Acumulado", format="%.2f%%"),
                    "Ingreso_Estimado": st.column_config.NumberColumn("Ingreso Estimado", format="$ %,.0f"),
                },
            )
        with tc2:
            st.markdown("**Participación por Planta de Producción**")
            st.dataframe(
                get_tabla_plantas(df_filt),
                hide_index=True,
                column_config={
                    "Demanda_Total": st.column_config.NumberColumn("Demanda Total", format="%,d"),
                    "Participacion_pct": st.column_config.NumberColumn("Participación", format="%.2f%%"),
                    "Ingreso_Estimado": st.column_config.NumberColumn("Ingreso Estimado", format="$ %,.0f"),
                },
            )

        st.markdown("---")
        # 4. Tendencias y Estacionalidad
        st.markdown("### 4. Tendencias y Estacionalidad")
        e_col1, e_col2 = st.columns([1.0, 1.15])
        with e_col1:
            st.plotly_chart(plot_indice_estacional(df_filt))
        with e_col2:
            modo_hm = st.radio(
                "Vista del Heatmap Estacional",
                options=["Producto vs Mes", "Año vs Mes"],
                horizontal=True,
                key="desc_hm_modo",
            )
            st.plotly_chart(plot_heatmap_estacionalidad(df_filt, modo=modo_hm))

        st.markdown("#### Descomposición Clásica de Series de Tiempo (`seasonal_decompose`, período = 12)")
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            dc_prod = st.selectbox(
                "Producto a descomponer",
                options=["Todos"] + [p for p in PRODUCTOS_ORDENADOS if p in sel_productos],
                index=1 if sel_productos else 0,
                key="dc_prod",
            )
        with dc2:
            dc_cli = st.selectbox(
                "Cliente a descomponer",
                options=["Todos"] + [c for c in CLIENTES_ORDENADOS if c in sel_clientes],
                index=0,
                key="dc_cli",
            )
        with dc3:
            dc_mod = st.selectbox(
                "Modelo de descomposición",
                options=["additive", "multiplicative"],
                index=0,
                key="dc_mod",
            )

        fig_decomp, decomp_info = plot_descomposicion_serie(
            df_filt,
            producto=dc_prod,
            cliente=dc_cli,
            modelo=dc_mod,
        )
        st.plotly_chart(fig_decomp)
        if decomp_info:
            di1, di2, di3, di4 = st.columns(4)
            di1.metric("Serie Analizada", decomp_info["serie"])
            di2.metric("Períodos Continuos", decomp_info["periodos"])
            di3.metric("Mes Pico Estacional", decomp_info["mes_pico_estacional"])
            di4.metric("Mes Valle Estacional", decomp_info["mes_valle_estacional"])

        st.markdown("---")
        # 5. Variabilidad de la demanda
        st.markdown("### 5. Variabilidad de la Demanda y Matriz ABC–XYZ")
        dim_box = st.radio(
            "Dimensión del Boxplot",
            options=["Producto", "Cliente"],
            horizontal=True,
            key="desc_box_dim",
        )
        st.plotly_chart(plot_boxplots_demanda(df_filt, dimension=dim_box))
        st.plotly_chart(plot_matriz_abc_xyz(df_filt))

        st.markdown("#### Tabla Estadística Completa (Media, Mediana, σ, IQR, CV% e Ingreso)")
        nivel_stats = st.radio(
            "Nivel de agregación estadística",
            options=["50 Series (Producto - Cliente)", "10 Productos (Agregado Mensual)"],
            horizontal=True,
            key="desc_stats_lvl",
        )
        por_arg = "Producto" if nivel_stats.startswith("10") else "Producto-Cliente"
        tabla_stats = get_tabla_estadistica_series(df_filt, por=por_arg)

        st.dataframe(
            tabla_stats,
            hide_index=True,
            height=400,
            column_config={
                "Demanda_Total": st.column_config.NumberColumn("Demanda Total", format="%,d"),
                "Participacion_pct": st.column_config.NumberColumn("Part. %", format="%.2f%%"),
                "Media": st.column_config.NumberColumn("Media (μ)", format="%,.2f"),
                "Mediana": st.column_config.NumberColumn("Mediana", format="%,.2f"),
                "Desv_Est": st.column_config.NumberColumn("Desv. Est. (σ)", format="%,.2f"),
                "IQR": st.column_config.NumberColumn("IQR", format="%,.2f"),
                "CV_pct": st.column_config.NumberColumn("CV (%)", format="%.2f%%"),
                "Ingreso_Estimado": st.column_config.NumberColumn("Ingreso Estimado", format="$ %,.0f"),
            },
        )
        st.download_button(
            label="⬇️ Descargar Tabla Estadística (CSV)",
            data=tabla_stats.to_csv(index=False).encode("utf-8"),
            file_name=f"estadisticas_demanda_{por_arg.lower()}.csv",
            mime="text/csv",
        )


# ==============================================================================
# PESTAÑA 4: FASE 3 — MODELOS DE PRONÓSTICO Y SERIES TEMPORALES
# ==============================================================================
with tab4:
    st.subheader("1. Configuración de Hiperparámetros y Validación Walk-Forward")
    st.markdown(
        """
        <div class="section-card">
        <b>Esquema de Validación Temporal Estricta (50 series independientes):</b>
        Entrenamiento inicial con los primeros <b>30 meses</b> (<code>2023-11</code> a <code>2026-04</code>)
        y validación <b>Walk-Forward 1 paso adelante</b> sobre los últimos <b>6 meses</b>
        (<code>2026-05</code> a <code>2026-10</code>), garantizando cero fuga de información.
        Posteriormente se reentrena con los <b>36 meses completos</b> para proyectar <b>Noviembre 2026 (t=37)</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    hp1, hp2, hp3, hp4, hp5 = st.columns(5)
    with hp1:
        ma_window = st.slider(
            "Ventana Promedio Móvil (k)",
            min_value=2,
            max_value=6,
            value=3,
            key="f3_ma_k",
        )
    with hp2:
        hw_seasonal = st.selectbox(
            "Estacionalidad Holt-Winters",
            options=["add", "mul"],
            index=0,
            format_func=lambda v: "Aditiva ('add')" if v == "add" else "Multiplicativa ('mul')",
            key="f3_hw_seas",
        )
    with hp3:
        mlp_lags = st.slider(
            "Rezagos Ventana MLP (W)",
            min_value=3,
            max_value=6,
            value=4,
            key="f3_mlp_lags",
        )
    with hp4:
        mlp_max_iter = st.slider(
            "Épocas MLP (max_iter)",
            min_value=100,
            max_value=300,
            value=200,
            step=50,
            key="f3_mlp_iter",
        )
    with hp5:
        criterio_sel = st.selectbox(
            "Criterio de Selección",
            options=["MAPE", "RMSE", "MAE"],
            index=0,
            key="f3_criterio",
        )

    forecast_res = ejecutar_pronosticos_cacheados(
        df_clean=df_clean,
        ma_window=ma_window,
        hw_seasonal=hw_seasonal,
        mlp_lags=mlp_lags,
        mlp_max_iter=mlp_max_iter,
        criterio=criterio_sel,
        exportar_csv=(uploaded_file is None),
    )

    st.markdown("---")
    st.subheader("2. Visor Interactivo por Serie (Producto × Cliente)")
    vs1, vs2, vs3 = st.columns([1.0, 1.0, 1.2])
    with vs1:
        prod_f3 = st.selectbox("Producto", options=PRODUCTOS_ORDENADOS, index=0, key="f3_prod")
    with vs2:
        cli_f3 = st.selectbox("Cliente", options=CLIENTES_ORDENADOS, index=0, key="f3_cli")

    detalle_serie = forecast_res["series_details"][(prod_f3, cli_f3)]
    ganador_serie = detalle_serie["winner"]
    with vs3:
        st.metric(
            "Modelo Ganador en esta Serie",
            ganador_serie,
            f"Pronóstico Nov-2026: {detalle_serie['nov_preds'][ganador_serie]:,} uds",
        )

    st.plotly_chart(plot_series_forecast_comparison(detalle_serie, prod_f3, cli_f3))

    st.markdown("#### Comparativa de Desempeño en Validación (6 meses) y Pronóstico Nov-2026")
    cols_cards = st.columns(4)
    for col_card, model_name in zip(cols_cards, MODEL_NAMES):
        m_vals = detalle_serie["metrics"][model_name]
        pred_nov = detalle_serie["nov_preds"][model_name]
        es_ganador = model_name == ganador_serie
        badge = " 🏆 Ganador" if es_ganador else ""
        with col_card:
            st.markdown(f"**{model_name}{badge}**")
            st.metric(
                label="Pronóstico Nov-2026",
                value=f"{pred_nov:,} uds",
                delta=f"{criterio_sel}: {m_vals[criterio_sel]:.2f}{'%' if criterio_sel == 'MAPE' else ''}",
                delta_color="normal" if es_ganador else "off",
            )
            st.caption(
                f"MAE: `{m_vals['MAE']:.2f}` | RMSE: `{m_vals['RMSE']:.2f}` | MAPE: `{m_vals['MAPE']:.2f}%`"
            )

    st.markdown("---")
    st.subheader("3. Ficha Técnica: Arquitectura MLP y Selección de Estacionalidad")
    ft1, ft2 = st.columns(2)
    with ft1:
        st.markdown(
            f"""
            <div class="section-card">
            <b>Arquitectura Red Neuronal Multicapa (MLPRegressor):</b><br>
            • <b>Ventana Deslizante (W={mlp_lags}):</b> <code>X_t = [D_{{t-{mlp_lags}}}, ..., D_{{t-1}}] → Y_t = D_t</code><br>
            • <b>Escalamiento sin fuga:</b> <code>StandardScaler</code> ajustado únicamente sobre el conjunto de entrenamiento en cada paso.<br>
            • <b>Capas Ocultas:</b> <code>(32, 16)</code> neuronas con activación <code>ReLU</code>.<br>
            • <b>Optimización:</b> <code>Adam</code> (retropropagación MSE, <code>alpha=0.01</code>, <code>max_iter={mlp_max_iter}</code>, <code>random_state=42</code>).
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ft2:
        st.markdown(
            """
            <div class="section-card">
            <b>Justificación Holt-Winters Aditivo vs. Multiplicativo:</b><br>
            • <b>Variante Aditiva (<code>seasonal='add'</code>, por defecto):</b> Recomendada cuando la amplitud de los ciclos estacionales anuales (<code>period=12</code>) se mantiene relativamente constante en unidades absolutas a lo largo del tiempo.<br>
            • <b>Variante Multiplicativa (<code>seasonal='mul'</code>):</b> Útil cuando la variación estacional crece proporcionalmente con el nivel de tendencia; requiere demanda estrictamente positiva (garantizada por la Fase 2).
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PESTAÑA 5: FASE 4 — EVALUACIÓN, SELECCIÓN AUTOMÁTICA Y MATRIZ D_{p,c}
# ==============================================================================
with tab5:
    df_metrics_50 = forecast_res["df_metrics"]
    df_forecasts_50 = forecast_res["df_forecasts"]
    df_prod_summary = forecast_res["df_product_summary"]

    st.subheader("1. Indicadores Globales del Pronóstico — Noviembre 2026")
    dem_total_nov = int(df_forecasts_50["Demanda_Pronosticada_Nov2026"].sum())
    ing_total_nov = float(df_forecasts_50["Ingreso_Pronosticado_Nov2026"].sum())
    mape_global = float(df_metrics_50["MAPE_Ganador"].mean())
    rmse_global = float(df_metrics_50["RMSE_Ganador"].mean())
    conteo_wins = df_metrics_50["Modelo_Ganador"].value_counts()
    modelo_top = str(conteo_wins.index[0])
    wins_top = int(conteo_wins.iloc[0])

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Demanda Total Nov-2026 (∑ D_pc)", f"{dem_total_nov:,} uds", "50 series Producto × Cliente")
    g2.metric("Ingreso Esperado Nov-2026", f"${ing_total_nov / 1e6:,.1f} M", f"${ing_total_nov:,.0f} COP")
    g3.metric("MAPE Promedio Ganadores", f"{mape_global:.2f}%", f"RMSE promedio: {rmse_global:.2f} uds")
    g4.metric("Modelo Más Ganador", modelo_top, f"{wins_top} de 50 series ({wins_top * 2}%)")

    st.markdown("---")
    st.subheader(f"2. Tabla Resumen Oficial por Producto (Métrica: {criterio_sel}) y Distribución de Victorias")
    r_col1, r_col2 = st.columns([1.35, 1.0])

    with r_col1:
        fmt_suffix = "%" if criterio_sel == "MAPE" else ""
        st.dataframe(
            df_prod_summary,
            hide_index=True,
            height=395,
            column_config={
                "Promedio móvil": st.column_config.NumberColumn(f"Promedio móvil ({criterio_sel})", format=f"%.2f{fmt_suffix}"),
                "Holt": st.column_config.NumberColumn(f"Holt ({criterio_sel})", format=f"%.2f{fmt_suffix}"),
                "Holt-Winters": st.column_config.NumberColumn(f"Holt-Winters ({criterio_sel})", format=f"%.2f{fmt_suffix}"),
                "MLP": st.column_config.NumberColumn(f"MLP ({criterio_sel})", format=f"%.2f{fmt_suffix}"),
                "Demanda_Total_Nov2026": st.column_config.NumberColumn("Demanda Nov-2026", format="%,d"),
            },
        )
        st.caption(
            f"El error por producto corresponde al promedio simple de `{criterio_sel}` entre sus 5 clientes."
        )

    with r_col2:
        st.plotly_chart(plot_winner_distribution_donut(df_metrics_50))

    st.markdown("---")
    st.subheader("3. Matriz de Demanda Pronosticada D_{p,c} (10 × 5) — Entrada para Fase 5 (PuLP)")
    st.plotly_chart(plot_demand_matrix_heatmap(df_forecasts_50))

    st.markdown("---")
    st.subheader("4. Detalle Completo de las 50 Series y Exportación")
    st.dataframe(
        df_metrics_50,
        hide_index=True,
        height=400,
        column_config={
            "Precio_Mediano": st.column_config.NumberColumn("Precio Mediano", format="$ %,.0f"),
            "MAPE_Promedio_movil": st.column_config.NumberColumn("MAPE PM", format="%.2f%%"),
            "MAPE_Holt": st.column_config.NumberColumn("MAPE Holt", format="%.2f%%"),
            "MAPE_Holt_Winters": st.column_config.NumberColumn("MAPE HW", format="%.2f%%"),
            "MAPE_MLP": st.column_config.NumberColumn("MAPE MLP", format="%.2f%%"),
            "MAE_Ganador": st.column_config.NumberColumn("MAE Ganador", format="%.2f"),
            "RMSE_Ganador": st.column_config.NumberColumn("RMSE Ganador", format="%.2f"),
            "MAPE_Ganador": st.column_config.NumberColumn("MAPE Ganador", format="%.2f%%"),
            "Demanda_Pronosticada_Nov2026": st.column_config.NumberColumn("Demanda Nov-2026", format="%,d"),
            "Ingreso_Pronosticado_Nov2026": st.column_config.NumberColumn("Ingreso Nov-2026", format="$ %,.0f"),
        },
    )

    ex1, ex2, ex3, ex4 = st.columns(4)
    with ex1:
        st.download_button(
            label="⬇️ `pronosticos_nov2026.csv`",
            data=convertir_df_a_csv_bytes(df_forecasts_50),
            file_name="pronosticos_nov2026.csv",
            mime="text/csv",
            type="primary",
            width="stretch",
        )
    with ex2:
        st.download_button(
            label="⬇️ `pronosticos_nov2026.xlsx`",
            data=convertir_df_a_excel_bytes(df_forecasts_50, sheet_name="Pronosticos_Nov2026"),
            file_name="pronosticos_nov2026.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )
    with ex3:
        st.download_button(
            label="⬇️ `metricas_modelos_50series.csv`",
            data=convertir_df_a_csv_bytes(df_metrics_50),
            file_name="metricas_modelos_50series.csv",
            mime="text/csv",
            width="stretch",
        )
    with ex4:
        st.download_button(
            label="⬇️ `metricas_modelos_50series.xlsx`",
            data=convertir_df_a_excel_bytes(df_metrics_50, sheet_name="Metricas_50_Series"),
            file_name="metricas_modelos_50series.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )


# ==============================================================================
# PESTAÑA 6: FASE 5 — MODELO DE OPTIMIZACIÓN MATEMÁTICA (PuLP)
# ==============================================================================
with tab6:
    st.subheader("1. Integración Predictiva → Prescriptiva")
    st.markdown(
        """
        <div class="section-card">
        <b>Flujo de integración:</b> Demanda histórica → ETL → Pronósticos → Selección del mejor modelo
        → <b>Matriz D<sub>p,c</sub> (Fase 4)</b> → + Costos + Capacidades → <b>Modelo de programación lineal (PuLP)</b>
        → Plan óptimo de <b>Producción + Trasbordo + Distribución</b> con <b>costo mínimo</b>.<br>
        Horizonte de planeación único: <b>Noviembre 2026</b> (1 mes). Unidades: <b>uds/mes</b> y <b>COP/ud</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    red1, red2 = st.columns(2)
    with red1:
        st.markdown("**Nodos de trasbordo (ubicación de referencia)**")
        st.dataframe(
            pd.DataFrame({"Nodo": NODOS, "Ubicación": [UBICACION_NODOS[n] for n in NODOS]}),
            hide_index=True,
        )
    with red2:
        st.markdown("**Empresas demandantes (ubicación de referencia)**")
        st.dataframe(
            pd.DataFrame(
                {"Cliente": CLIENTES_ORDENADOS, "Ciudad": [UBICACION_CLIENTES[c] for c in CLIENTES_ORDENADOS]}
            ),
            hide_index=True,
        )
    st.caption(
        "Las ubicaciones, distancias, flotas y factores de costo son supuestos de diseño del grupo "
        "(el enunciado pide construir los parámetros con unidades y justificación)."
    )

    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Formulación Matemática del Modelo")
    with st.expander("📐 Ver conjuntos, variables, función objetivo y restricciones", expanded=False):
        st.markdown(
            "**Conjuntos:** $P$ = productos (10), $C$ = clientes (5), $F$ = plantas (4), $N$ = nodos de trasbordo (2).\n\n"
            "**Variables de decisión (uds/mes, continuas ≥ 0):** $Q_{fp}$ producción · $X_{fnp}$ envío planta→nodo · "
            "$Y_{ncp}$ envío nodo→cliente · $S_{pc}$ demanda no satisfecha (solo si se permiten faltantes)."
        )
        st.markdown("**Función objetivo (COP):**")
        st.latex(
            r"\min Z = \sum_{f,p} CP_{fp}\,Q_{fp} + \sum_{f,n,p} CT_{fnp}\,X_{fnp} + \sum_{n,c,p} CD_{ncp}\,Y_{ncp}"
            r" \;\left[+ \sum_{p,c} Pen_{pc}\,S_{pc}\right]"
        )
        st.markdown("**Restricciones:**")
        st.latex(r"\text{R1 Demanda: } \sum_{n} Y_{ncp} \;[+\,S_{pc}] = D_{pc} \quad \forall p,c")
        st.latex(r"\text{R2 Capacidad de producción: } Q_{fp} \le Cap_{fp} \quad \forall f,p")
        st.latex(r"\text{R3 Balance en plantas: } Q_{fp} = \sum_{n} X_{fnp} \quad \forall f,p")
        st.latex(r"\text{R4 Balance en nodos: } \sum_{f} X_{fnp} = \sum_{c} Y_{ncp} \quad \forall n,p")
        st.latex(r"\text{R5 Capacidad de nodos: } \sum_{f,p} X_{fnp} \le CapN_{n} \quad \forall n")
        st.latex(
            r"\text{R6 Transporte: } \sum_{p} X_{fnp} \le CapT_{fn} \;\; \forall f,n \qquad "
            r"\sum_{p} Y_{ncp} \le CapD_{nc} \;\; \forall n,c"
        )
        st.latex(r"\text{R7 No negatividad: } Q_{fp},\, X_{fnp},\, Y_{ncp} \,[,\,S_{pc}] \ge 0")
        st.caption(
            "El término entre corchetes solo se activa con la opción 'Permitir faltantes'. Sin ella, si la capacidad "
            "no alcanza, el modelo se reporta como NO FACTIBLE en lugar de mostrar una solución que incumpla restricciones."
        )

    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Parámetros del Modelo (unidades y justificación)")

    with st.expander("⚙️ Ajustar supuestos de costos y capacidades", expanded=False):
        sp1, sp2, sp3, sp4 = st.columns(4)
        with sp1:
            pct_cp = st.slider("Costo producción (% del precio)", 40, 70, int(SUPUESTOS_DEFAULT["pct_costo_produccion"] * 100), key="f5_pct_cp")
            recargo = st.slider("Recargo planta no asignada (%)", 0, 40, int(SUPUESTOS_DEFAULT["recargo_planta_secundaria"] * 100), key="f5_recargo")
        with sp2:
            margen = st.slider("Margen capacidad sobre pico histórico (%)", 0, 30, int(SUPUESTOS_DEFAULT["margen_capacidad"] * 100), key="f5_margen")
            pct_sec = st.slider("Capacidad plantas no asignadas (%)", 0, 50, int(SUPUESTOS_DEFAULT["pct_capacidad_secundaria"] * 100), key="f5_pct_sec")
        with sp3:
            tarifa_t = st.number_input("Tarifa troncal (COP/ud·km)", 0.5, 10.0, SUPUESTOS_DEFAULT["tarifa_troncal"], 0.1, key="f5_tt")
            tarifa_d = st.number_input("Tarifa distribución (COP/ud·km)", 0.5, 15.0, SUPUESTOS_DEFAULT["tarifa_distribucion"], 0.1, key="f5_td")
        with sp4:
            cap_norte = st.number_input("CapN Nodo Norte (uds/mes)", 0, 20000, int(CAP_NODO_DEFAULT["Nodo Norte"]), 100, key="f5_capn_n")
            cap_centro = st.number_input("CapN Nodo Centro (uds/mes)", 0, 20000, int(CAP_NODO_DEFAULT["Nodo Centro"]), 100, key="f5_capn_c")

    supuestos_sel = {
        **SUPUESTOS_DEFAULT,
        "pct_costo_produccion": pct_cp / 100,
        "recargo_planta_secundaria": recargo / 100,
        "margen_capacidad": margen / 100,
        "pct_capacidad_secundaria": pct_sec / 100,
        "tarifa_troncal": float(tarifa_t),
        "tarifa_distribucion": float(tarifa_d),
    }
    cap_nodo_sel = {"Nodo Norte": float(cap_norte), "Nodo Centro": float(cap_centro)}

    oc1, oc2 = st.columns([1.2, 1.0])
    with oc1:
        permitir_faltantes = st.toggle(
            "Permitir faltantes con penalización (1,5 × precio por unidad no atendida)",
            value=False,
            key="f5_faltantes",
        )
    with oc2:
        st.caption(
            "Por defecto toda la demanda debe atenderse (R1 con igualdad). Active los faltantes solo si quiere "
            "ver qué parte de la demanda no se puede cubrir cuando la capacidad no alcanza."
        )

    res_opt = resolver_optimizacion_cacheada(
        df_forecasts=forecast_res["df_forecasts"],
        df_clean=df_clean,
        supuestos_items=tuple(sorted(supuestos_sel.items())),
        cap_nodo_items=tuple(sorted(cap_nodo_sel.items())),
        factor_demanda=1.0,
        permitir_faltantes=permitir_faltantes,
    )
    par_opt = res_opt["parametros"]
    tabs_par = tablas_parametros(par_opt)

    pt0, pt1, pt2, pt3, pt4, pt5 = st.tabs(
        ["📋 Supuestos", "D_pc Demanda", "Cap_fp y CP_fp Producción", "CT_fnp Transporte", "CD_ncp Distribución", "CapN · CapT · CapD"]
    )
    with pt0:
        st.dataframe(pd.DataFrame(TABLA_SUPUESTOS), hide_index=True)
    with pt1:
        st.markdown("Demanda pronosticada **D_pc (uds/mes)** — modelo ganador por serie (Fase 4).")
        st.dataframe(tabs_par["D"], column_config={c: st.column_config.NumberColumn(format="%,.0f") for c in tabs_par["D"].columns})
    with pt2:
        cpa, cpb = st.columns(2)
        with cpa:
            st.markdown("**Cap_fp — Capacidad de producción (uds/mes)**")
            st.dataframe(tabs_par["Cap"])
        with cpb:
            st.markdown("**CP_fp — Costo unitario de producción (COP/ud)**")
            st.dataframe(tabs_par["CP"], column_config={c: st.column_config.NumberColumn(format="$ %,.0f") for c in tabs_par["CP"].columns})
    with pt3:
        st.markdown("**CT_fnp — Costo unitario planta → nodo (COP/ud)** = tarifa troncal × km + seguro de carga")
        st.dataframe(tabs_par["CT"], column_config={c: st.column_config.NumberColumn(format="$ %,.0f") for c in tabs_par["CT"].columns})
    with pt4:
        st.markdown("**CD_ncp — Costo unitario nodo → cliente (COP/ud)** = tarifa distribución × km + seguro + cross-docking")
        st.dataframe(tabs_par["CD"], column_config={c: st.column_config.NumberColumn(format="$ %,.0f") for c in tabs_par["CD"].columns})
    with pt5:
        st.markdown("**CapN_n — Capacidad de procesamiento de los nodos**")
        st.dataframe(tabs_par["CapN"], hide_index=True)
        cap_a, cap_b = st.columns(2)
        with cap_a:
            st.markdown("**CapT_fn — Capacidad de transporte planta → nodo**")
            st.dataframe(tabs_par["CapT"], hide_index=True)
        with cap_b:
            st.markdown("**CapD_nc — Capacidad de transporte nodo → cliente**")
            st.dataframe(tabs_par["CapD"], hide_index=True)
        st.caption(f"Camión de referencia: {int(par_opt.supuestos['uds_por_camion']):,} uds. Arcos con 0 camiones no están permitidos.")

    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("4. Solución Óptima")
    st.caption(
        f"Modelo de programación lineal: {res_opt['n_variables']} variables y {res_opt['n_restricciones']} restricciones · "
        "Solver: CBC (incluido en PuLP)."
    )

    if not res_opt["optimo"]:
        st.error(
            f"**Estado del solver: {res_opt['estado']}.** Con los parámetros actuales NO existe un plan que atienda toda "
            "la demanda respetando las capacidades. No se muestra ninguna solución para no presentar un plan que incumpla restricciones."
        )
        st.markdown("**Posibles cuellos de botella detectados:**")
        for msg in res_opt["diagnostico"]:
            st.markdown(f"- {msg}")
        st.info("Ajuste las capacidades en ⚙️ o active **'Permitir faltantes'** para cuantificar la demanda no atendida.")
    else:
        df_Q_opt, df_X_opt, df_Y_opt, df_S_opt = res_opt["df_Q"], res_opt["df_X"], res_opt["df_Y"], res_opt["df_S"]
        util = res_opt["utilizacion"]
        faltante_total = float(df_S_opt["Faltante"].sum())

        st.success(f"**Estado del solver: Óptimo.** Costo total mínimo Z = **${res_opt['Z']:,.0f} COP**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Costo Total Mínimo (Z)", f"${res_opt['Z'] / 1e6:,.1f} M", f"${res_opt['Z']:,.0f} COP", delta_color="off")
        k2.metric("Costo de Producción", f"${res_opt['costo_produccion'] / 1e6:,.1f} M", f"{res_opt['costo_produccion'] / res_opt['Z'] * 100:.1f}% de Z", delta_color="off")
        k3.metric("Transporte Planta → Nodo", f"${res_opt['costo_transporte'] / 1e6:,.1f} M", f"{res_opt['costo_transporte'] / res_opt['Z'] * 100:.1f}% de Z", delta_color="off")
        k4.metric("Distribución Nodo → Cliente", f"${res_opt['costo_distribucion'] / 1e6:,.1f} M", f"{res_opt['costo_distribucion'] / res_opt['Z'] * 100:.1f}% de Z", delta_color="off")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Unidades Producidas", f"{df_Q_opt['Cantidad'].sum():,.0f} uds")
        k6.metric("Demanda Atendida", f"{df_S_opt['Atendido'].sum():,.0f} uds", f"de {df_S_opt['Demanda'].sum():,.0f} uds", delta_color="off")
        k7.metric("Demanda No Satisfecha", f"{faltante_total:,.0f} uds", "Penalización: " + f"${res_opt['costo_penalizacion']:,.0f}", delta_color="inverse" if faltante_total > 0 else "off")
        k8.metric("Costo Promedio por Unidad", f"${res_opt['Z'] / max(df_S_opt['Atendido'].sum(), 1):,.0f}")

        if faltante_total > 0.5:
            st.warning(
                f"Se dejan de atender **{faltante_total:,.0f} uds**: la capacidad de la red no alcanza para toda la demanda."
            )
            st.dataframe(df_S_opt[df_S_opt["Faltante"] > 0.5], hide_index=True)

        # ---------------- Respuestas a las preguntas del enunciado ----------------
        st.markdown("#### ¿Qué producir, cuánto y en cuál planta?  (Q_fp)")
        st.plotly_chart(plot_produccion_por_planta(df_Q_opt))
        piv_Q = df_Q_opt.pivot(index="Producto", columns="Planta", values="Cantidad").loc[PRODUCTOS_ORDENADOS, PLANTAS_OFICIALES]
        piv_Q["Total"] = piv_Q.sum(axis=1)
        st.dataframe(piv_Q, column_config={c: st.column_config.NumberColumn(format="%,.0f") for c in piv_Q.columns})

        fuera = df_Q_opt[(df_Q_opt["Cantidad"] > 0.5) & (df_Q_opt["Planta"] != df_Q_opt["Producto"].map(MAPA_PRODUCTO_PLANTA))]
        if not fuera.empty:
            lista = ", ".join(f"{r.Producto} en {r.Planta} ({r.Cantidad:,.0f} uds)" for r in fuera.itertuples())
            st.info(
                f"**Hallazgo:** el modelo decide producir fuera de la planta asignada: {lista}. Aunque producir allí cuesta "
                f"un {recargo}% más, el ahorro en transporte (o la falta de capacidad de la planta asignada) lo justifica."
            )

        st.markdown("#### ¿Qué cantidad enviar por cada nodo de trasbordo?  (X_fnp)")
        tx1, tx2 = st.columns([1.0, 1.2])
        with tx1:
            res_x = df_X_opt.groupby(["Planta", "Nodo"])["Cantidad"].sum().unstack("Nodo").reindex(index=PLANTAS_OFICIALES, columns=NODOS)
            res_x["Total"] = res_x.sum(axis=1)
            st.dataframe(res_x, column_config={c: st.column_config.NumberColumn(format="%,.0f") for c in res_x.columns})
        with tx2:
            piv_X = df_X_opt.pivot_table(index="Producto", columns="Nodo", values="Cantidad", aggfunc="sum").reindex(index=PRODUCTOS_ORDENADOS, columns=NODOS)
            st.dataframe(piv_X, column_config={c: st.column_config.NumberColumn(format="%,.0f") for c in piv_X.columns})

        st.markdown("#### ¿Cuánto distribuir a cada cliente?  (Y_ncp)")
        ty1, ty2 = st.columns([1.0, 1.2])
        with ty1:
            res_y = df_Y_opt.groupby(["Nodo", "Cliente"])["Cantidad"].sum().unstack("Cliente").reindex(index=NODOS, columns=CLIENTES_ORDENADOS)
            st.dataframe(res_y, column_config={c: st.column_config.NumberColumn(format="%,.0f") for c in res_y.columns})
        with ty2:
            piv_Y = df_Y_opt.pivot_table(index="Producto", columns="Cliente", values="Cantidad", aggfunc="sum").reindex(index=PRODUCTOS_ORDENADOS, columns=CLIENTES_ORDENADOS)
            st.dataframe(piv_Y, column_config={c: st.column_config.NumberColumn(format="%,.0f") for c in piv_Y.columns})

        st.markdown("#### Red de flujo óptima y costo total mínimo")
        sk1, sk2 = st.columns([1.6, 1.0])
        with sk1:
            st.plotly_chart(plot_sankey_red(res_opt))
        with sk2:
            st.plotly_chart(plot_composicion_costos(res_opt))

        # ---------------- Utilización ----------------
        st.markdown("---")
        st.subheader("5. Utilización de Capacidades y Cuellos de Botella")
        u1, u2 = st.columns(2)
        with u1:
            st.plotly_chart(plot_utilizacion(util["plantas"], "Planta", "Utilización de plantas (sobre capacidad total por planta)"))
        with u2:
            st.plotly_chart(plot_utilizacion(util["nodos"], "Nodo", "Utilización de nodos de trasbordo (CapN)"))
        u3, u4 = st.columns(2)
        arcos_t = util["arcos_planta_nodo"][util["arcos_planta_nodo"]["Permitido"]].copy()
        arcos_t["Arco"] = arcos_t["Planta"] + " → " + arcos_t["Nodo"]
        arcos_d = util["arcos_nodo_cliente"][util["arcos_nodo_cliente"]["Permitido"]].copy()
        arcos_d["Arco"] = arcos_d["Nodo"] + " → " + arcos_d["Cliente"]
        with u3:
            st.plotly_chart(plot_utilizacion(arcos_t, "Arco", "Utilización de transporte planta → nodo (CapT)"))
        with u4:
            st.plotly_chart(plot_utilizacion(arcos_d, "Arco", "Utilización de distribución nodo → cliente (CapD)"))

        saturados = df_Q_opt[(df_Q_opt["Capacidad"] > 0) & (df_Q_opt["Utilizacion_%"] >= 99.5)]
        cuellos = [f"{r.Producto} en {r.Planta}" for r in saturados.itertuples()]
        cuellos += util["nodos"].loc[util["nodos"]["Utilizacion_%"] >= 99.5, "Nodo"].tolist()
        arcos_todos = pd.concat([arcos_t, arcos_d], ignore_index=True)
        cuellos += arcos_todos.loc[arcos_todos["Utilizacion_%"] >= 99.5, "Arco"].tolist()
        if cuellos:
            st.warning("**Restricciones activas (capacidad al 100 %):** " + "; ".join(cuellos) + ".")
        else:
            st.info("Ninguna capacidad está al 100 %: la red tiene holgura para la demanda pronosticada.")
        st.caption(
            "La utilización por planta divide la producción entre la suma de capacidades de todos sus productos "
            "(incluye la capacidad de respaldo para productos no asignados)."
        )

        # ---------------- Verificación ----------------
        st.markdown("---")
        st.subheader("6. Verificación Independiente de Restricciones")
        st.dataframe(res_opt["verificacion"], hide_index=True)
        if res_opt["verificacion"]["Cumple"].all():
            st.success("La solución cumple las 7 restricciones del modelo (tolerancia 0,0001 uds).")
        else:
            st.error("La solución NO cumple alguna restricción; revise los parámetros.")

        # ---------------- Exportación ----------------
        st.markdown("---")
        resumen_costos = pd.DataFrame(
            {
                "Componente": ["Producción", "Transporte planta→nodo", "Distribución nodo→cliente", "Penalización faltantes", "Costo total Z"],
                "COP": [res_opt["costo_produccion"], res_opt["costo_transporte"], res_opt["costo_distribucion"], res_opt["costo_penalizacion"], res_opt["Z"]],
            }
        )
        hojas = {
            "Resumen_Costos": resumen_costos,
            "Produccion_Q": df_Q_opt,
            "Trasbordo_X": df_X_opt[df_X_opt["Cantidad"] > 0.5],
            "Distribucion_Y": df_Y_opt[df_Y_opt["Cantidad"] > 0.5],
            "Demanda_Atendida": df_S_opt,
            "Util_Plantas": util["plantas"],
            "Util_Nodos": util["nodos"],
            "Util_Planta_Nodo": util["arcos_planta_nodo"],
            "Util_Nodo_Cliente": util["arcos_nodo_cliente"],
            "Verificacion": res_opt["verificacion"],
            "Param_D": tabs_par["D"],
            "Param_Cap": tabs_par["Cap"],
            "Param_CP": tabs_par["CP"],
            "Param_CT": tabs_par["CT"],
            "Param_CD": tabs_par["CD"],
            "Param_CapN": tabs_par["CapN"],
            "Param_CapT": tabs_par["CapT"],
            "Param_CapD": tabs_par["CapD"],
            "Supuestos": pd.DataFrame(TABLA_SUPUESTOS),
        }
        st.download_button(
            label="⬇️ Descargar plan óptimo y parámetros (`plan_optimo_nov2026.xlsx`)",
            data=convertir_varios_df_a_excel_bytes(hojas),
            file_name="plan_optimo_nov2026.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            width="stretch",
        )


# ==============================================================================
# Fase 6 — Análisis de escenarios
# ==============================================================================
with tab7:
    st.header("Fase 6 — Análisis de escenarios de demanda")
    st.caption("Optimista = demanda −10 % · Base = pronóstico de la Fase 4 · Pesimista = demanda +10 %. "
               "Mismos costos y capacidades de la Fase 5; solo cambia D_pc.")

    esc = correr_escenarios_cacheado(
        forecast_res["df_forecasts"], df_clean,
        tuple(sorted(supuestos_sel.items())), tuple(sorted(cap_nodo_sel.items())),
    )
    df_res = tabla_resumen(esc)

    # --- Alertas de factibilidad (no mostrar óptimos falsos) ---
    for n, e in esc.items():
        if e["con_faltantes"]:
            falt = e["mostrado"]["df_S"]["Faltante"].sum() if e["mostrado"]["optimo"] else float("nan")
            st.error(f"**{n}:** el modelo NO es factible atendiendo el 100 % de la demanda. "
                     f"Resuelto con faltantes penalizados: quedan **{falt:,.0f} uds** sin atender.")
            for m in e["estricto"].get("diagnostico", []):
                st.markdown(f"- {m}")

    st.subheader("1. Comparación general")
    fmt = {c: st.column_config.NumberColumn(format="%,.0f") for c in df_res.columns
           if "COP" in c or "uds" in c}
    fmt["Δ Z vs Base (%)"] = st.column_config.NumberColumn(format="%+.1f%%")
    st.dataframe(df_res, hide_index=True, column_config=fmt)
    st.plotly_chart(plot_costo_escenarios(df_res))

    st.subheader("2. Producción y utilización de plantas")
    df_pl = tabla_plantas(esc)
    c1, c2 = st.columns(2)
    c1.plotly_chart(plot_produccion_planta_esc(df_pl))
    c2.plotly_chart(plot_utilizacion_esc(df_pl, "Planta", "Utilización de plantas por escenario"))

    st.subheader("3. Utilización de nodos de trasbordo")
    st.plotly_chart(plot_utilizacion_esc(tabla_nodos(esc), "Nodo", "Utilización de nodos por escenario"))

    st.subheader("4. Distribución a clientes")
    df_cl = tabla_clientes(esc)
    st.plotly_chart(plot_clientes_esc(df_cl))
    st.dataframe(df_cl, hide_index=True)

    st.subheader("5. Restricciones activas (capacidad al 100 %)")
    for n in ["Optimista", "Base", "Pesimista"]:
        act = restricciones_activas(esc[n]["mostrado"])
        st.markdown(f"**{n}:** " + ("; ".join(act) if act else "ninguna, hay holgura."))

    st.subheader("6. Lectura de resultados")
    st.info("Escriba aquí 3–4 conclusiones propias (qué escenario encarece más, qué planta/nodo se satura, "
            "si aparecen faltantes y qué decisión recomendaría).")
