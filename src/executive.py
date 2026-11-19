"""Fase 7: indicadores verificables que conectan las Fases 1 a 6.

Esta capa no depende de Streamlit ni de PuLP: facilita las pruebas y evita
atribuir una producción o una distribución óptima a una solución infactible.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.etl import CLIENTES_ORDENADOS, PRODUCTOS_ORDENADOS

COLUMNAS_HIST = {"Fecha", "Producto", "Cliente", "Demanda", "Planta", "Precio"}
COLUMNAS_PRON = {
    "Producto", "Cliente", "Modelo_Ganador", "Demanda_Pronosticada_Nov2026"
}


def validar_cadena_datos(df_clean: pd.DataFrame, df_forecasts: pd.DataFrame,
                         df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Comprueba presencia, unicidad, continuidad y validez de la integración."""
    pruebas: list[dict[str, Any]] = []

    def anotar(etapa: str, nombre: str, cumple: bool, detalle: str) -> None:
        pruebas.append({"Fase": etapa, "Control": nombre,
                        "Estado": "Correcto" if cumple else "Revisar",
                        "Detalle": detalle})

    columnas_ok = COLUMNAS_HIST.issubset(df_clean.columns)
    anotar("1–2", "Campos de la base histórica", columnas_ok,
           "Fecha, Producto, Cliente, Demanda, Planta y Precio")
    if columnas_ok:
        original = df_clean.copy()
        fechas = pd.to_datetime(original["Fecha"], errors="coerce")
        claves = ["Fecha", "Producto", "Cliente"]
        validas = original[claves].notna().all(axis=1) & fechas.notna()
        anotar("1–2", "Datos válidos y sin duplicados",
               bool(validas.all() and not original.duplicated(subset=claves).any()
                    and pd.to_numeric(original["Demanda"], errors="coerce").notna().all()
                    and (pd.to_numeric(original["Demanda"], errors="coerce") >= 0).all()),
               f"{len(original):,} registros; {int(original[claves].duplicated().sum())} claves repetidas")
        esperado = {(p, c) for p in PRODUCTOS_ORDENADOS for c in CLIENTES_ORDENADOS}
        actual = set(zip(original["Producto"], original["Cliente"]))
        counts = original.groupby(["Producto", "Cliente"]).size()
        meses = pd.DatetimeIndex(fechas.dropna().unique()).sort_values()
        consecutivos = bool(len(meses) == 36 and
                            meses.equals(pd.date_range(meses.min(), periods=36, freq="MS")))
        panel_ok = bool(actual == esperado and len(original) == 1800
                        and len(counts) == 50 and counts.eq(36).all() and consecutivos)
        anotar("1–2", "Panel balanceado 10 × 5 × 36", panel_ok,
               f"{len(counts)} series; {len(meses)} meses distintos")
    else:
        anotar("1–2", "Panel balanceado 10 × 5 × 36", False,
               "No se puede comprobar: faltan columnas históricas")

    pron_ok = COLUMNAS_PRON.issubset(df_forecasts.columns)
    anotar("3–4", "Campos de pronóstico", pron_ok,
           "Producto, Cliente, Modelo_Ganador y Demanda_Pronosticada_Nov2026")
    if pron_ok:
        forecast_pairs = set(zip(df_forecasts["Producto"], df_forecasts["Cliente"]))
        demanda = pd.to_numeric(df_forecasts["Demanda_Pronosticada_Nov2026"], errors="coerce")
        anotar("3–4", "50 pronósticos vinculables", bool(
            len(df_forecasts) == 50 and not df_forecasts.duplicated(["Producto", "Cliente"]).any()
            and forecast_pairs == {(p, c) for p in PRODUCTOS_ORDENADOS for c in CLIENTES_ORDENADOS}
            and demanda.notna().all() and np.isfinite(demanda).all() and (demanda >= 0).all()),
            f"{len(df_forecasts)} filas; total = {demanda.sum():,.0f} uds")
    if {"Producto", "Cliente", "Modelo_Ganador"}.issubset(df_metrics.columns) and pron_ok:
        m = df_metrics[["Producto", "Cliente", "Modelo_Ganador"]].copy()
        p = df_forecasts[["Producto", "Cliente", "Modelo_Ganador"]].copy()
        iguales = (len(m) == len(p) == 50 and
                   not m.duplicated(["Producto", "Cliente"]).any() and
                   (m.merge(p, on=["Producto", "Cliente"], how="outer",
                            suffixes=("_evaluado", "_pronostico"), indicator=True)
                    .pipe(lambda d: d["_merge"].eq("both").all()
                          and d["Modelo_Ganador_evaluado"].eq(d["Modelo_Ganador_pronostico"]).all())))
        anotar("3–4", "Ganadores consistentes", bool(iguales),
               "Coincidencia entre evaluación y pronóstico por producto-cliente")
    else:
        anotar("3–4", "Ganadores consistentes", False,
               "Faltan columnas para verificar modelos ganadores")
    return pd.DataFrame(pruebas)


def tabla_demanda_atencion(df_forecasts: pd.DataFrame, res_opt: dict) -> pd.DataFrame:
    """Pronóstico más unidades efectivamente atendidas, si hay solución válida."""
    columnas = ["Producto", "Cliente", "Modelo_Ganador", "Demanda_Pronosticada_Nov2026"]
    tabla = df_forecasts[columnas].copy().rename(columns={
        "Demanda_Pronosticada_Nov2026": "Demanda pronosticada (uds)",
        "Modelo_Ganador": "Modelo ganador",
    })
    if "MAPE_Ganador" in df_forecasts.columns:
        tabla["MAPE (%)"] = df_forecasts["MAPE_Ganador"].to_numpy()
    verificable = (res_opt.get("optimo", False)
                   and res_opt.get("verificacion") is not None
                   and bool(res_opt["verificacion"]["Cumple"].all()))
    if verificable and {"Producto", "Cliente", "Atendido", "Faltante"}.issubset(
        res_opt["df_S"].columns
    ):
        destino = res_opt["df_S"][["Producto", "Cliente", "Atendido", "Faltante"]].copy()
        destino = destino.rename(columns={"Atendido": "Atendido (uds)", "Faltante": "Faltante (uds)"})
        tabla = tabla.merge(destino, on=["Producto", "Cliente"], how="left", validate="one_to_one")
        tabla["Cobertura (%)"] = np.where(
            tabla["Demanda pronosticada (uds)"] > 0,
            100 * tabla["Atendido (uds)"] / tabla["Demanda pronosticada (uds)"], 100.0,
        )
    else:
        # No se muestran 'unidades atendidas = 0': se desconoce la solución, no la demanda.
        tabla["Atendido (uds)"] = np.nan
        tabla["Faltante (uds)"] = np.nan
        tabla["Cobertura (%)"] = np.nan
    return tabla


def indicadores_ejecutivos(df_clean: pd.DataFrame, df_forecasts: pd.DataFrame,
                            df_metrics: pd.DataFrame, res_opt: dict) -> dict[str, Any]:
    """Indicadores con estado de factibilidad explícito."""
    d_total = float(df_forecasts["Demanda_Pronosticada_Nov2026"].sum())
    d_anterior = float(df_clean.loc[
        pd.to_datetime(df_clean["Fecha"]) == pd.to_datetime(df_clean["Fecha"]).max(), "Demanda"
    ].sum())
    out = {
        "demanda_pronosticada": d_total,
        "demanda_ultimo_mes": d_anterior,
        "variacion_vs_ultimo_mes_pct": (d_total / d_anterior - 1) * 100 if d_anterior else np.nan,
        "series": int(df_forecasts[["Producto", "Cliente"]].drop_duplicates().shape[0]),
        "mape_promedio_pct": float(df_metrics["MAPE_Ganador"].mean())
        if "MAPE_Ganador" in df_metrics else np.nan,
        "estado_optimizacion": res_opt.get("estado", "No disponible"),
        "optimo": bool(res_opt.get("optimo", False)),
        "costo_cop": np.nan,
        "cobertura_pct": np.nan,
        "faltante_unidades": np.nan,
        "restricciones_verificadas": False,
    }
    if out["optimo"] and res_opt.get("verificacion") is not None and \
            bool(res_opt["verificacion"]["Cumple"].all()):
        suma_atendida = float(res_opt["df_S"]["Atendido"].sum())
        suma_faltante = float(res_opt["df_S"]["Faltante"].sum())
        out.update({
            "costo_cop": float(res_opt["Z"]),
            "cobertura_pct": 100 * suma_atendida / d_total if d_total else 100.0,
            "faltante_unidades": suma_faltante,
            "restricciones_verificadas": bool(res_opt["verificacion"]["Cumple"].all()),
        })
    return out


def conclusiones_escenarios(esc: dict, resumen: pd.DataFrame) -> list[str]:
    """Genera mensajes basados en estados, costos, faltantes y utilizaciones reales."""
    notas: list[str] = []
    for nombre in ("Optimista", "Base", "Pesimista"):
        if nombre not in esc:
            continue
        caso = esc[nombre]
        estricto = caso.get("estricto", {})
        mostrado = caso.get("mostrado", {})
        if not estricto.get("optimo", False):
            estado = estricto.get("estado", "Desconocido")
            if estado in ("Solver no disponible", "Error al resolver"):
                notas.append(f"{nombre}: no hay resultado del optimizador ({estado}).")
            elif mostrado.get("optimo", False):
                falt = float(mostrado["df_S"]["Faltante"].sum())
                notas.append(
                    f"{nombre}: la demanda completa no es factible; al permitir faltantes "
                    f"penalizados quedan {falt:,.0f} unidades sin atender."
                )
            elif estado == "Infeasible":
                notas.append(f"{nombre}: no hay solución sin faltantes y tampoco se obtuvo un plan penalizado válido.")
            else:
                notas.append(f"{nombre}: no se dispone de una solución verificable ({estado}).")
    if {"Optimista", "Base", "Pesimista"}.issubset(esc):
        if all(esc[n]["estricto"].get("optimo", False) for n in ("Optimista", "Base", "Pesimista")):
            costs = {n: float(esc[n]["estricto"]["Z"]) for n in ("Optimista", "Base", "Pesimista")}
            notas.append(
                "Con cobertura completa, el costo total operativo es "
                f"${costs['Optimista']:,.0f} COP (optimista), "
                f"${costs['Base']:,.0f} COP (base) y "
                f"${costs['Pesimista']:,.0f} COP (pesimista)."
            )
        else:
            if any(esc[n].get("con_faltantes", False) for n in ("Optimista", "Base", "Pesimista")):
                notas.append(
                    "Los costos de los escenarios con faltantes incluyen una penalización. "
                    "No deben compararse como si fueran costos de una operación que atiende toda la demanda."
                )
            else:
                notas.append("No se dispone de costos óptimos comparables para los tres escenarios.")
    for nombre in ("Base", "Pesimista", "Optimista"):
        r = esc.get(nombre, {}).get("mostrado", {})
        if not r.get("optimo", False):
            continue
        u = r.get("utilizacion", {})
        nombres_saturados = []
        for llave, columna in (("plantas", "Planta"), ("nodos", "Nodo")):
            df = u.get(llave)
            if df is not None and not df.empty:
                nombres_saturados += [str(x) for x in df.loc[df["Utilizacion_%"] >= 99.5, columna]]
        if nombres_saturados:
            notas.append(f"{nombre}: capacidades prácticamente saturadas en "
                         + ", ".join(nombres_saturados[:5]) + ".")
        else:
            notas.append(f"{nombre}: no se detectó una planta ni un nodo al 99,5 % o más de utilización.")
        break
    if not notas:
        notas.append("No hay resultados verificables para extraer conclusiones de los escenarios.")
    return notas
