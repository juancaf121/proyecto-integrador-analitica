"""
Fase 6 — Análisis de escenarios de demanda.

Reutiliza el modelo de la Fase 5 (construir_parametros / resolver_modelo) cambiando solo
el factor de demanda:
    Optimista : demanda 10 % inferior  (factor 0.90)
    Base      : demanda pronosticada   (factor 1.00)
    Pesimista : demanda 10 % superior  (factor 1.10)

Regla del enunciado: nunca mostrar una solución "óptima" que incumpla restricciones.
Por eso, si un escenario es infactible con R1 en igualdad, se informa como INFACTIBLE y
se resuelve una segunda vez permitiendo faltantes penalizados solo para cuantificar
qué demanda no se alcanza a cubrir (queda marcado como tal).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.etl import CLIENTES_ORDENADOS, PLANTAS_OFICIALES
from src.optimization import NODOS, construir_parametros, resolver_modelo

ESCENARIOS: dict[str, float] = {"Optimista": 0.90, "Base": 1.00, "Pesimista": 1.10}
ORDEN = list(ESCENARIOS)
COLOR_ESC = {"Optimista": "#59a14f", "Base": "#0f4c81", "Pesimista": "#d62728"}


def correr_escenarios(
    df_forecasts: pd.DataFrame,
    df_clean: pd.DataFrame,
    supuestos: dict | None = None,
    cap_nodo: dict | None = None,
    factores: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Resuelve cada escenario. Devuelve {nombre: {factor, estricto, mostrado, con_faltantes}}."""
    factores = factores or ESCENARIOS
    out = {}
    for nombre, f in factores.items():
        par = construir_parametros(df_forecasts, df_clean, supuestos, f, cap_nodo)
        estricto = resolver_modelo(par, permitir_faltantes=False)
        if estricto["optimo"] or estricto["estado"] != "Infeasible":
            # Un error técnico o un estado indeterminado no equivale a infactibilidad.
            mostrado, con_falt = estricto, False
        else:
            mostrado, con_falt = resolver_modelo(par, permitir_faltantes=True), True
        out[nombre] = {"factor": f, "estricto": estricto, "mostrado": mostrado, "con_faltantes": con_falt}
    return out


def _estado(e: dict) -> str:
    estado_estricto = e["estricto"].get("estado", "Desconocido")
    if e["estricto"].get("optimo", False):
        return "Óptimo, 100 % atendido"
    if estado_estricto in ("Solver no disponible", "Error al resolver"):
        return estado_estricto
    if not e["con_faltantes"]:
        return estado_estricto
    return "INFACTIBLE sin faltantes" if e["mostrado"]["optimo"] else "INFACTIBLE (sin solución)"


def restricciones_activas(res: dict, tol: float = 99.5) -> list[str]:
    """Capacidades al 100 % en la solución mostrada."""
    if not res.get("optimo"):
        return []
    u, q = res["utilizacion"], res["df_Q"]
    act = [f"Producción {r.Producto} en {r.Planta}" for r in q[(q["Capacidad"] > 0) & (q["Utilizacion_%"] >= tol)].itertuples()]
    act += [f"Nodo {r.Nodo}".replace("Nodo Nodo", "Nodo") for r in u["nodos"][u["nodos"]["Utilizacion_%"] >= tol].itertuples()]
    for k, a, b in (("arcos_planta_nodo", "Planta", "Nodo"), ("arcos_nodo_cliente", "Nodo", "Cliente")):
        d = u[k][(u[k]["Permitido"]) & (u[k]["Utilizacion_%"] >= tol)]
        act += [f"Arco {getattr(r, a)} → {getattr(r, b)}" for r in d.itertuples()]
    return act


# ------------------------------------------------------------------ tablas comparativas
def tabla_resumen(esc: dict[str, dict]) -> pd.DataFrame:
    filas = []
    for n in ORDEN:
        e, r = esc[n], esc[n]["mostrado"]
        if not r["optimo"]:
            filas.append({"Escenario": n, "Demanda": f"{e['factor']:.0%}", "Estado": _estado(e)})
            continue
        dem, falt = r["df_S"]["Demanda"].sum(), r["df_S"]["Faltante"].sum()
        filas.append({
            "Escenario": n, "Demanda": f"{e['factor']:.0%}", "Estado": _estado(e),
            "Demanda total (uds)": dem, "Atendida (uds)": dem - falt, "Faltante (uds)": falt,
            "Producción (COP)": r["costo_produccion"], "Transporte (COP)": r["costo_transporte"],
            "Distribución (COP)": r["costo_distribucion"], "Penalización (COP)": r["costo_penalizacion"],
            "Costo total Z (COP)": r["Z"], "Costo/ud atendida (COP)": r["Z"] / max(dem - falt, 1),
        })
    df = pd.DataFrame(filas)
    if "Costo total Z (COP)" not in df:
        df["Costo total Z (COP)"] = float("nan")
    base = df.loc[df["Escenario"] == "Base", "Costo total Z (COP)"]
    if len(base) and pd.notna(base.iloc[0]):
        df["Δ Z vs Base (%)"] = (df["Costo total Z (COP)"] / base.iloc[0] - 1) * 100
    return df


def _largo(esc, extractor) -> pd.DataFrame:
    partes = []
    for n in ORDEN:
        if esc[n]["mostrado"]["optimo"]:
            d = extractor(esc[n]["mostrado"]).copy()
            d.insert(0, "Escenario", n)
            partes.append(d)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def tabla_plantas(esc):  # producción y utilización por planta
    return _largo(esc, lambda r: r["utilizacion"]["plantas"])


def tabla_nodos(esc):
    return _largo(esc, lambda r: r["utilizacion"]["nodos"])


def tabla_clientes(esc):  # unidades distribuidas y demanda por cliente
    def f(r):
        y = r["df_Y"].groupby("Cliente")["Cantidad"].sum().reindex(CLIENTES_ORDENADOS)
        s = r["df_S"].groupby("Cliente")[["Demanda", "Faltante"]].sum().reindex(CLIENTES_ORDENADOS)
        return pd.DataFrame({"Cliente": CLIENTES_ORDENADOS, "Demanda": s["Demanda"].values,
                             "Distribuido": y.values, "Faltante": s["Faltante"].values})
    return _largo(esc, f)


# ------------------------------------------------------------------ gráficos
def _layout(fig, titulo, h=380):
    fig.update_layout(title=titulo, template="plotly_white", height=h, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_costo_escenarios(df_res: pd.DataFrame) -> go.Figure:
    d = df_res.dropna(subset=["Costo total Z (COP)"])
    comp = ["Producción (COP)", "Transporte (COP)", "Distribución (COP)", "Penalización (COP)"]
    fig = go.Figure()
    for c, col in zip(comp, ["#0f4c81", "#1b7895", "#f28e2b", "#d62728"]):
        fig.add_bar(x=d["Escenario"], y=d[c], name=c.replace(" (COP)", ""), marker_color=col)
    fig.update_layout(barmode="stack", yaxis_title="COP")
    return _layout(fig, "Costo total por escenario (apilado por componente)")


def plot_produccion_planta_esc(df_pl: pd.DataFrame) -> go.Figure:
    fig = px.bar(df_pl, x="Planta", y="Cantidad", color="Escenario", barmode="group",
                 color_discrete_map=COLOR_ESC, category_orders={"Planta": PLANTAS_OFICIALES, "Escenario": ORDEN},
                 text_auto=",.0f", labels={"Cantidad": "Unidades producidas"})
    return _layout(fig, "Producción por planta y escenario")


def plot_utilizacion_esc(df: pd.DataFrame, x: str, titulo: str) -> go.Figure:
    fig = px.bar(df, x=x, y="Utilizacion_%", color="Escenario", barmode="group",
                 color_discrete_map=COLOR_ESC, category_orders={"Escenario": ORDEN}, text_auto=".1f")
    fig.add_hline(y=100, line_dash="dash", line_color="#d62728")
    fig.update_layout(yaxis_title="Utilización (%)")
    return _layout(fig, titulo)


def plot_clientes_esc(df_cl: pd.DataFrame) -> go.Figure:
    fig = px.bar(df_cl, x="Cliente", y="Distribuido", color="Escenario", barmode="group",
                 color_discrete_map=COLOR_ESC, category_orders={"Escenario": ORDEN}, text_auto=",.0f",
                 labels={"Distribuido": "Unidades distribuidas"})
    return _layout(fig, "Unidades distribuidas por cliente y escenario")
