"""
Módulo ETL (Extract, Transform, Load) para la Plataforma Inteligente de Planeación
de Producción y Distribución.

Transforma el archivo crudo `demanda_historica_cruda.csv` en un panel balanceado,
determinista y auditable de exactamente 1.800 observaciones:
10 productos × 5 clientes × 36 meses (2023-11-01 a 2026-10-01).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

# ==============================================================================
# Constantes oficiales de la red logística
# ==============================================================================
PRODUCTOS_ORDENADOS: list[str] = [f"P{i}" for i in range(1, 11)]
CLIENTES_ORDENADOS: list[str] = [f"Cliente {i}" for i in range(1, 6)]
PLANTAS_OFICIALES: list[str] = ["Bogotá", "Medellín", "Cali", "Bucaramanga"]

MAPA_PRODUCTO_PLANTA: dict[str, str] = {
    "P1": "Bogotá",
    "P5": "Bogotá",
    "P9": "Bogotá",
    "P2": "Medellín",
    "P6": "Medellín",
    "P10": "Medellín",
    "P3": "Cali",
    "P7": "Cali",
    "P4": "Bucaramanga",
    "P8": "Bucaramanga",
}

MESES_ES: dict[str, str] = {
    "ene": "01",
    "feb": "02",
    "mar": "03",
    "abr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "ago": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dic": "12",
}

FECHAS_IMPOSIBLES_MAP: dict[str, str] = {
    "30/02/2025": "2025-02-01",
    "29/02/2025": "2025-02-01",
    "2025-02-29": "2025-02-01",
    "2025-02-30": "2025-02-01",
    "30/02/2026": "2026-02-01",
    "29/02/2026": "2026-02-01",
    "2026-02-29": "2026-02-01",
    "2026-02-30": "2026-02-01",
}

HORIZONTE_INICIO = "2023-11-01"
HORIZONTE_FIN = "2026-10-01"
NULL_TOKENS = {"", "nan", "null", "n/a", "na", "s/d", "nd", "-", "none"}


# ==============================================================================
# Funciones auxiliares de limpieza y normalización
# ==============================================================================
def clean_fecha(val: Any) -> pd.Timestamp | pd.NaT:
    """Normaliza múltiples formatos de fecha en español/ISO/DD-MM-YYYY a YYYY-MM-01."""
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip().lower()
    if s in NULL_TOKENS:
        return pd.NaT

    if s in FECHAS_IMPOSIBLES_MAP:
        return pd.Timestamp(FECHAS_IMPOSIBLES_MAP[s])

    for mes_str, mes_num in MESES_ES.items():
        if mes_str in s:
            m1 = re.match(r"^([a-z]{3})[-/\s](\d{2,4})$", s)
            if m1 and m1.group(1) == mes_str:
                yr = int(m1.group(2))
                if yr < 100:
                    yr += 2000
                return pd.Timestamp(f"{yr:04d}-{mes_num}-01")

            m2 = re.match(r"^(\d{1,2})[-/\s]([a-z]{3})[-/\s](\d{2,4})$", s)
            if m2 and m2.group(2) == mes_str:
                yr = int(m2.group(3))
                if yr < 100:
                    yr += 2000
                return pd.Timestamp(f"{yr:04d}-{mes_num}-01")

    if re.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$", s):
        dt = pd.to_datetime(s, format="ISO8601", errors="coerce")
        if pd.notna(dt):
            return pd.Timestamp(year=dt.year, month=dt.month, day=1)

    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if pd.notna(dt):
        return pd.Timestamp(year=dt.year, month=dt.month, day=1)

    return pd.NaT


def clean_producto(val: Any) -> float | str:
    """Estandariza variantes de producto ('p6', 'P 2', 'P-09') a 'P1'..'P10'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in NULL_TOKENS:
        return np.nan
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    return f"P{int(m.group(1))}"


def clean_cliente(val: Any) -> float | str:
    """Estandariza variantes de cliente ('Cte. 1', 'Cliente2') a 'Cliente 1'..'Cliente 5'."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in NULL_TOKENS:
        return np.nan
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    return f"Cliente {int(m.group(1))}"


def clean_planta(val: Any) -> float | str:
    """Homologa las variantes textuales a las 4 plantas oficiales de producción."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in NULL_TOKENS:
        return np.nan
    if any(k in s for k in ("bog", "bgta")):
        return "Bogotá"
    if "med" in s:
        return "Medellín"
    if "cal" in s:
        return "Cali"
    if any(k in s for k in ("buc", "b/manga", "bmanga")):
        return "Bucaramanga"
    return np.nan


def parse_demanda(val: Any) -> float:
    """Convierte cadenas de demanda con sufijos ('u', 'und', 'unidades') a float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in NULL_TOKENS:
        return np.nan
    s = re.sub(r"(unidades|und|u)\b", "", s).strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_precio(val: Any) -> float:
    """Convierte precios monetarios ('COP', '$', ',00', separadores de miles) a float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.lower() in NULL_TOKENS:
        return np.nan
    s = re.sub(r"(?i)cop|\$", "", s).strip()
    s = re.sub(r",00$", "", s).strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _read_raw_source(source: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Lee un DataFrame crudo desde ruta CSV/Excel o DataFrame en memoria."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _ordenar_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena numéricamente por Producto (P1..P10), Cliente (1..5) y Fecha ascendente."""
    out = df.copy()
    out["_prod_num"] = out["Producto"].astype(str).str.extract(r"(\d+)").astype(int)
    out["_cli_num"] = out["Cliente"].astype(str).str.extract(r"(\d+)").astype(int)
    out = (
        out.sort_values(["_prod_num", "_cli_num", "Fecha"])
        .drop(columns=["_prod_num", "_cli_num"])
        .reset_index(drop=True)
    )
    return out


# ==============================================================================
# Pipeline ETL principal
# ==============================================================================
def run_etl_pipeline(
    raw_source: Union[str, Path, pd.DataFrame] = "demanda_historica_cruda.csv",
    output_csv_path: Union[str, Path, None] = "data/demanda_limpia_1800.csv",
    audit_json_path: Union[str, Path, None] = "data/auditoria_etl.json",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Ejecuta las 6 etapas del pipeline ETL y retorna:
    - df_clean: Panel balanceado limpio de 1.800 observaciones.
    - df_pre_interp: Panel de 1.800 celdas antes de remover outliers e interpolar.
    - audit_report: Diccionario serializable con métricas y detalles de cada etapa.
    """
    # --------------------------------------------------------------------------
    # Etapa 1: Auditoría inicial sobre los datos crudos
    # --------------------------------------------------------------------------
    df_raw = _read_raw_source(raw_source)

    filas_crudas = int(len(df_raw))
    filas_vacias = int(df_raw.isna().all(axis=1).sum())
    duplicados_exactos = int(df_raw.duplicated().sum())
    nulos_iniciales = {col: int(cnt) for col, cnt in df_raw.isna().sum().items()}
    formatos_fecha_unicos = int(df_raw["Fecha"].dropna().nunique())
    variantes_producto = int(df_raw["Producto"].dropna().nunique())
    variantes_cliente = int(df_raw["Cliente"].dropna().nunique())
    variantes_planta = int(df_raw["Planta"].dropna().nunique())

    df = df_raw.copy()

    # --------------------------------------------------------------------------
    # Etapa 2: Normalización de Fecha (a YYYY-MM-01)
    # --------------------------------------------------------------------------
    fecha_raw_norm = df["Fecha"].astype(str).str.strip().str.lower()
    mask_fechas_imposibles = df["Fecha"].notna() & fecha_raw_norm.isin(FECHAS_IMPOSIBLES_MAP)
    fechas_imposibles_detalle = [
        {
            "Fila_Original": int(idx) + 2,
            "Fecha_Original": str(df_raw.loc[idx, "Fecha"]),
            "Fecha_Corregida": FECHAS_IMPOSIBLES_MAP[str(df_raw.loc[idx, "Fecha"]).strip().lower()],
        }
        for idx in df[mask_fechas_imposibles].index
    ]

    df["Fecha"] = df["Fecha"].apply(clean_fecha)

    # --------------------------------------------------------------------------
    # Etapa 3: Estandarización de Categorías (Producto, Cliente, Planta)
    # --------------------------------------------------------------------------
    df["Producto"] = df["Producto"].apply(clean_producto)
    df["Cliente"] = df["Cliente"].apply(clean_cliente)
    df["Planta"] = df["Planta"].apply(clean_planta)

    # --------------------------------------------------------------------------
    # Etapa 4: Limpieza Numérica de Demanda y Precio
    # --------------------------------------------------------------------------
    dem_parsed = df["Demanda"].apply(parse_demanda)
    mask_dem_anomala = dem_parsed.notna() & (dem_parsed <= 0)
    demanda_neg_cero_detalle = [
        {
            "Fila_Original": int(idx) + 2,
            "Fecha": df.loc[idx, "Fecha"].strftime("%Y-%m-%d") if pd.notna(df.loc[idx, "Fecha"]) else None,
            "Producto": df.loc[idx, "Producto"] if pd.notna(df.loc[idx, "Producto"]) else None,
            "Cliente": df.loc[idx, "Cliente"] if pd.notna(df.loc[idx, "Cliente"]) else None,
            "Valor_Crudo": str(df_raw.loc[idx, "Demanda"]),
            "Valor_Numerico": float(dem_parsed.loc[idx]),
            "Tipo": "Negativo" if dem_parsed.loc[idx] < 0 else "Cero",
        }
        for idx in df[mask_dem_anomala].index
    ]
    df["Demanda"] = dem_parsed.where(dem_parsed > 0, np.nan)

    prec_parsed = df["Precio"].apply(parse_precio)
    mask_prec_anomalo = prec_parsed.notna() & ((prec_parsed <= 0) | (prec_parsed > 100000))
    precios_atipicos_detalle = [
        {
            "Fila_Original": int(idx) + 2,
            "Fecha": df.loc[idx, "Fecha"].strftime("%Y-%m-%d") if pd.notna(df.loc[idx, "Fecha"]) else None,
            "Producto": df.loc[idx, "Producto"] if pd.notna(df.loc[idx, "Producto"]) else None,
            "Cliente": df.loc[idx, "Cliente"] if pd.notna(df.loc[idx, "Cliente"]) else None,
            "Precio_Crudo": str(df_raw.loc[idx, "Precio"]),
            "Precio_Numerico": float(prec_parsed.loc[idx]),
            "Motivo": "Precio <= 0" if prec_parsed.loc[idx] <= 0 else "Precio > 100.000 (error digitación)",
        }
        for idx in df[mask_prec_anomalo].index
    ]
    df["Precio"] = prec_parsed.where((prec_parsed > 0) & (prec_parsed <= 100000), np.nan)

    # --------------------------------------------------------------------------
    # Etapa 5: Recuperación de Llaves, Consolidación y Reindexación (10 x 5 x 36)
    # --------------------------------------------------------------------------
    med_precio_planta_prod = df.groupby(["Planta", "Producto"])["Precio"].median()
    mask_recup = df["Producto"].isna() & df["Planta"].notna() & df["Precio"].notna()
    productos_recuperados_detalle: list[dict[str, Any]] = []

    for idx in df[mask_recup].index:
        pl = df.loc[idx, "Planta"]
        pr = float(df.loc[idx, "Precio"])
        if pl in med_precio_planta_prod.index.get_level_values(0):
            candidates = med_precio_planta_prod.loc[pl]
            best_prod = str((candidates - pr).abs().idxmin())
            med_ref = float(candidates.loc[best_prod])
            df.loc[idx, "Producto"] = best_prod
            productos_recuperados_detalle.append(
                {
                    "Fila_Original": int(idx) + 2,
                    "Fecha": df.loc[idx, "Fecha"].strftime("%Y-%m-%d") if pd.notna(df.loc[idx, "Fecha"]) else None,
                    "Cliente": df.loc[idx, "Cliente"] if pd.notna(df.loc[idx, "Cliente"]) else None,
                    "Planta": pl,
                    "Precio": pr,
                    "Producto_Inferido": best_prod,
                    "Mediana_Referencia": med_ref,
                }
            )

    df_valid_keys = df.dropna(subset=["Fecha", "Producto", "Cliente"]).copy()
    filas_con_llaves_validas = int(len(df_valid_keys))
    filas_descartadas_llave_nula = int(filas_crudas - filas_con_llaves_validas)
    colisiones_llaves_duplicadas = int(
        df_valid_keys.duplicated(subset=["Fecha", "Producto", "Cliente"]).sum()
    )

    df_grouped = df_valid_keys.groupby(["Fecha", "Producto", "Cliente"], as_index=False).agg(
        {
            "Demanda": "median",
            "Precio": "median",
            "Planta": "first",
        }
    )
    combinaciones_unicas_validas = int(len(df_grouped))

    fechas_36 = pd.date_range(HORIZONTE_INICIO, HORIZONTE_FIN, freq="MS")
    idx_full = pd.MultiIndex.from_product(
        [fechas_36, PRODUCTOS_ORDENADOS, CLIENTES_ORDENADOS],
        names=["Fecha", "Producto", "Cliente"],
    )
    df_panel = (
        df_grouped.set_index(["Fecha", "Producto", "Cliente"])
        .reindex(idx_full)
        .reset_index()
    )
    df_panel = _ordenar_panel(df_panel)
    df_panel["Es_Celda_Faltante_Malla"] = df_panel["Demanda"].isna() & df_panel["Planta"].isna()
    celdas_reconstruidas_panel = int(len(idx_full) - combinaciones_unicas_validas)

    # --------------------------------------------------------------------------
    # Etapa 6: Detección de Outliers (3.0 × IQR) e Imputación Temporal por Serie
    # --------------------------------------------------------------------------
    df_pre_interp = df_panel[["Fecha", "Producto", "Cliente", "Demanda", "Planta", "Precio"]].copy()
    df_pre_interp = df_pre_interp.rename(
        columns={"Demanda": "Demanda_Observada", "Precio": "Precio_Observado"}
    )
    df_pre_interp["Limite_Inferior"] = np.nan
    df_pre_interp["Limite_Superior"] = np.nan
    df_pre_interp["Es_Outlier"] = False
    df_pre_interp["Es_Celda_Faltante"] = df_pre_interp["Demanda_Observada"].isna()

    outliers_detalle: list[dict[str, Any]] = []
    demanda_imputada_col = pd.Series(index=df_panel.index, dtype="int64")

    for (prod, cli), grp in df_panel.groupby(["Producto", "Cliente"], sort=False):
        s = grp["Demanda"]
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        upper = q3 + 3.0 * iqr
        lower = max(0.0, q1 - 3.0 * iqr)
        mask_out = (s > upper) | (s < lower)

        s_clean = s.mask(mask_out, np.nan)
        s_interp = s_clean.interpolate(method="linear").bfill().ffill().round().astype(int)

        df_pre_interp.loc[grp.index, "Limite_Inferior"] = round(lower, 2)
        df_pre_interp.loc[grp.index, "Limite_Superior"] = round(upper, 2)
        df_pre_interp.loc[grp.index, "Es_Outlier"] = mask_out.values

        for idx_row in grp[mask_out].index:
            outliers_detalle.append(
                {
                    "Fecha": df_panel.loc[idx_row, "Fecha"].strftime("%Y-%m-%d"),
                    "Producto": prod,
                    "Cliente": cli,
                    "Q1": round(q1, 2),
                    "Q3": round(q3, 2),
                    "IQR": round(iqr, 2),
                    "Limite_Inferior": round(lower, 2),
                    "Limite_Superior": round(upper, 2),
                    "Valor_Original": int(round(float(s.loc[idx_row]))),
                    "Valor_Imputado": int(s_interp.loc[idx_row]),
                }
            )

        demanda_imputada_col.loc[grp.index] = s_interp

    df_panel["Demanda"] = demanda_imputada_col.astype(int)
    df_panel["Planta"] = df_panel["Producto"].map(MAPA_PRODUCTO_PLANTA)
    df_panel["Precio"] = (
        df_panel.groupby("Producto")["Precio"]
        .transform(lambda x: x.fillna(x.median()))
        .round(2)
    )

    df_clean = df_panel[["Fecha", "Producto", "Cliente", "Demanda", "Planta", "Precio"]].copy()
    df_clean = _ordenar_panel(df_clean)
    df_pre_interp["Demanda_Limpia"] = df_clean["Demanda"].values

    nulos_finales = {col: int(cnt) for col, cnt in df_clean.isna().sum().items()}
    total_valores_imputados_demanda = int(df_pre_interp["Es_Celda_Faltante"].sum() + len(outliers_detalle))

    audit_report: dict[str, Any] = {
        "filas_crudas": filas_crudas,
        "filas_vacias": filas_vacias,
        "duplicados_exactos": duplicados_exactos,
        "nulos_iniciales": nulos_iniciales,
        "nulos_finales": nulos_finales,
        "formatos_fecha_unicos": formatos_fecha_unicos,
        "variantes_producto": variantes_producto,
        "variantes_cliente": variantes_cliente,
        "variantes_planta": variantes_planta,
        "fechas_imposibles_corregidas": len(fechas_imposibles_detalle),
        "fechas_imposibles_detalle": fechas_imposibles_detalle,
        "demanda_neg_cero_corregidos": len(demanda_neg_cero_detalle),
        "demanda_neg_cero_detalle": demanda_neg_cero_detalle,
        "precios_atipicos_corregidos": len(precios_atipicos_detalle),
        "precios_atipicos_detalle": precios_atipicos_detalle,
        "productos_recuperados": len(productos_recuperados_detalle),
        "productos_recuperados_detalle": productos_recuperados_detalle,
        "filas_con_llaves_validas": filas_con_llaves_validas,
        "filas_descartadas_llave_nula": filas_descartadas_llave_nula,
        "colisiones_llaves_duplicadas": colisiones_llaves_duplicadas,
        "combinaciones_unicas_validas": combinaciones_unicas_validas,
        "celdas_reconstruidas_panel": celdas_reconstruidas_panel,
        "outliers_detectados": len(outliers_detalle),
        "outliers_detalle": outliers_detalle,
        "total_celdas_demanda_imputadas": total_valores_imputados_demanda,
        "registros_limpios_finales": int(len(df_clean)),
        "horizonte_inicio": df_clean["Fecha"].min().strftime("%Y-%m-%d"),
        "horizonte_fin": df_clean["Fecha"].max().strftime("%Y-%m-%d"),
        "meses_continuos": int(df_clean["Fecha"].nunique()),
        "productos_unicos": int(df_clean["Producto"].nunique()),
        "clientes_unicos": int(df_clean["Cliente"].nunique()),
        "plantas_unicas": int(df_clean["Planta"].nunique()),
        "series_unicas": int(df_clean.groupby(["Producto", "Cliente"]).ngroups),
        "demanda_total_limpia": int(df_clean["Demanda"].sum()),
    }

    if output_csv_path is not None:
        out_csv = Path(output_csv_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_export = df_clean.copy()
        df_export["Fecha"] = df_export["Fecha"].dt.strftime("%Y-%m-%d")
        df_export.to_csv(out_csv, index=False, encoding="utf-8")

    if audit_json_path is not None:
        out_json = Path(audit_json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2, ensure_ascii=False)

    return df_clean, df_pre_interp, audit_report


def load_clean_data(csv_path: Union[str, Path] = "data/demanda_limpia_1800.csv") -> pd.DataFrame:
    """
    Carga la base de datos limpia lista para consumo en Analítica Descriptiva (Fase 2)
    y Modelos de Pronóstico (Fase 3).

    Garantiza:
    - `Fecha` como `datetime64[ns]`
    - `Demanda` como `int64`
    - `Precio` como `float64`
    - Ordenamiento determinista por Producto (P1..P10), Cliente (Cliente 1..5) y Fecha.
    """
    path = Path(csv_path)
    if not path.exists():
        df_clean, _, _ = run_etl_pipeline(output_csv_path=path)
        return df_clean

    df = pd.read_csv(path)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Demanda"] = df["Demanda"].astype(int)
    df["Precio"] = df["Precio"].astype(float)
    return _ordenar_panel(df)


if __name__ == "__main__":
    clean_df, _, audit = run_etl_pipeline()
    print(f"ETL completado: {len(clean_df)} registros limpios | Demanda total: {audit['demanda_total_limpia']:,}")
