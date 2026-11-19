"""
Fase 5 — Modelo de optimización matemática (Analítica Prescriptiva).

Modelo de programación lineal de costo mínimo para la red:
    4 plantas (F)  →  2 nodos de trasbordo (N)  →  5 clientes (C),  10 productos (P)

Decisiones:
    Q[f,p]   : unidades del producto p producidas en la planta f
    X[f,n,p] : unidades del producto p enviadas de la planta f al nodo n
    Y[n,c,p] : unidades del producto p enviadas del nodo n al cliente c
    S[p,c]   : (opcional) demanda no satisfecha, penalizada en la función objetivo

La demanda D[p,c] proviene de la Fase 4 (modelo ganador por serie, Nov-2026).
Todos los costos están en COP por unidad y las capacidades en unidades por mes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp

from src.etl import (
    CLIENTES_ORDENADOS,
    MAPA_PRODUCTO_PLANTA,
    PLANTAS_OFICIALES,
    PRODUCTOS_ORDENADOS,
)

# ==============================================================================
# 1. Conjuntos y supuestos de la red logística (definidos por el grupo)
# ==============================================================================
NODOS: list[str] = ["Nodo Norte", "Nodo Centro"]

# Ubicación de referencia de los nodos y clientes (supuesto de diseño del grupo).
UBICACION_NODOS: dict[str, str] = {
    "Nodo Norte": "Malambo (Atlántico) — atiende la Costa Caribe",
    "Nodo Centro": "Funza (Cundinamarca) — atiende el centro del país",
}
UBICACION_CLIENTES: dict[str, str] = {
    "Cliente 1": "Bogotá D.C.",
    "Cliente 2": "Barranquilla",
    "Cliente 3": "Cartagena",
    "Cliente 4": "Tunja",
    "Cliente 5": "Villavicencio",
}

# Distancias terrestres aproximadas por carretera (km).
DIST_PLANTA_NODO: dict[tuple[str, str], float] = {
    ("Bogotá", "Nodo Norte"): 1000, ("Bogotá", "Nodo Centro"): 25,
    ("Medellín", "Nodo Norte"): 700, ("Medellín", "Nodo Centro"): 415,
    ("Cali", "Nodo Norte"): 1120, ("Cali", "Nodo Centro"): 460,
    ("Bucaramanga", "Nodo Norte"): 560, ("Bucaramanga", "Nodo Centro"): 395,
}
DIST_NODO_CLIENTE: dict[tuple[str, str], float] = {
    ("Nodo Norte", "Cliente 1"): 1000, ("Nodo Centro", "Cliente 1"): 25,
    ("Nodo Norte", "Cliente 2"): 15, ("Nodo Centro", "Cliente 2"): 1000,
    ("Nodo Norte", "Cliente 3"): 125, ("Nodo Centro", "Cliente 3"): 1050,
    ("Nodo Norte", "Cliente 4"): 850, ("Nodo Centro", "Cliente 4"): 160,
    ("Nodo Norte", "Cliente 5"): 1110, ("Nodo Centro", "Cliente 5"): 140,
}

# Flota contratada por arco (camiones por mes). 0 = arco NO permitido.
CAMIONES_PLANTA_NODO: dict[tuple[str, str], int] = {
    ("Bogotá", "Nodo Norte"): 1, ("Bogotá", "Nodo Centro"): 6,
    ("Medellín", "Nodo Norte"): 2, ("Medellín", "Nodo Centro"): 3,
    ("Cali", "Nodo Norte"): 0, ("Cali", "Nodo Centro"): 2,
    ("Bucaramanga", "Nodo Norte"): 2, ("Bucaramanga", "Nodo Centro"): 1,
}
CAMIONES_NODO_CLIENTE: dict[tuple[str, str], int] = {
    ("Nodo Norte", "Cliente 1"): 1, ("Nodo Centro", "Cliente 1"): 4,
    ("Nodo Norte", "Cliente 2"): 2, ("Nodo Centro", "Cliente 2"): 1,
    ("Nodo Norte", "Cliente 3"): 2, ("Nodo Centro", "Cliente 3"): 1,
    ("Nodo Norte", "Cliente 4"): 1, ("Nodo Centro", "Cliente 4"): 2,
    ("Nodo Norte", "Cliente 5"): 0, ("Nodo Centro", "Cliente 5"): 1,
}

# Capacidad de procesamiento (cross-docking) de cada nodo, en unidades/mes.
CAP_NODO_DEFAULT: dict[str, float] = {"Nodo Norte": 5000, "Nodo Centro": 6500}

# Factor regional de costo de producción (energía, mano de obra, servicios).
FACTOR_COSTO_PLANTA: dict[str, float] = {
    "Bogotá": 1.03,
    "Medellín": 1.00,
    "Cali": 0.98,
    "Bucaramanga": 0.97,
}

# Supuestos escalares por defecto (editables desde la app).
SUPUESTOS_DEFAULT: dict[str, float] = {
    "pct_costo_produccion": 0.55,       # CP base = 55 % del precio mediano
    "recargo_planta_secundaria": 0.15,  # +15 % si el producto no es de la planta asignada
    "margen_capacidad": 0.05,           # Capacidad planta asignada = pico histórico × 1.05
    "pct_capacidad_secundaria": 0.25,   # Plantas no asignadas: 25 % de esa capacidad
    "tarifa_troncal": 2.2,              # COP/(ud·km) planta → nodo (camión completo)
    "tarifa_distribucion": 3.5,         # COP/(ud·km) nodo → cliente (menor consolidación)
    "pct_seguro_carga": 0.002,          # Seguro de carga: 0,2 % del valor por trayecto
    "costo_cross_docking": 120.0,       # COP/ud por manipulación en el nodo
    "uds_por_camion": 1200.0,           # Unidades que caben en un camión de 10 t
    "factor_penalizacion": 1.5,         # Faltante penalizado en 1,5 × precio
}

TABLA_SUPUESTOS: list[dict[str, str]] = [
    {"Parámetro": "D_pc", "Unidad": "uds/mes",
     "Construcción": "Pronóstico del modelo ganador (Fase 4) para cada Producto × Cliente, Nov-2026.",
     "Justificación": "Conecta la analítica predictiva con la prescriptiva: la demanda a cubrir es la pronosticada."},
    {"Parámetro": "Cap_fp", "Unidad": "uds/mes",
     "Construcción": "Planta asignada: pico mensual histórico del producto × (1 + margen). Otras plantas: un % de esa capacidad.",
     "Justificación": "La capacidad instalada se dimensiona sobre el máximo histórico; las demás plantas tienen líneas compartidas o maquila limitada."},
    {"Parámetro": "CP_fp", "Unidad": "COP/ud",
     "Construcción": "% del precio mediano del producto × factor regional de la planta × (1 + recargo si no es la planta asignada).",
     "Justificación": "El costo de producción guarda proporción con el precio; producir fuera de la planta especializada es más costoso."},
    {"Parámetro": "CT_fnp", "Unidad": "COP/ud",
     "Construcción": "Tarifa troncal × distancia(f,n) + seguro de carga (% del precio del producto).",
     "Justificación": "Transporte en camión completo: el costo crece con los km; el seguro depende del valor de la mercancía."},
    {"Parámetro": "CD_ncp", "Unidad": "COP/ud",
     "Construcción": "Tarifa de distribución × distancia(n,c) + seguro de carga + costo de cross-docking en el nodo.",
     "Justificación": "La distribución final consolida menos carga (tarifa mayor) e incluye la manipulación en el nodo."},
    {"Parámetro": "CapN_n", "Unidad": "uds/mes",
     "Construcción": "Capacidad de recibo/despacho de cada centro de trasbordo.",
     "Justificación": "Nodo Centro mayor (atiende Bogotá, Tunja y Villavicencio); Nodo Norte atiende la Costa Caribe."},
    {"Parámetro": "CapT_fn", "Unidad": "uds/mes",
     "Construcción": "Camiones contratados por mes × unidades por camión. 0 camiones = arco no permitido.",
     "Justificación": "Refleja la flota disponible; Cali → Nodo Norte no tiene ruta contratada por la distancia (1.120 km)."},
    {"Parámetro": "CapD_nc", "Unidad": "uds/mes",
     "Construcción": "Camiones contratados por mes × unidades por camión. 0 camiones = arco no permitido.",
     "Justificación": "Nodo Norte → Villavicencio no está permitido (1.110 km); se atiende solo desde el Nodo Centro."},
]


# ==============================================================================
# 2. Estructura de parámetros
# ==============================================================================
@dataclass
class ParametrosRed:
    D: dict[tuple[str, str], float]
    Cap: dict[tuple[str, str], float]
    CP: dict[tuple[str, str], float]
    CT: dict[tuple[str, str, str], float]
    CD: dict[tuple[str, str, str], float]
    CapN: dict[str, float]
    CapT: dict[tuple[str, str], float]
    CapD: dict[tuple[str, str], float]
    Penal: dict[tuple[str, str], float]
    precios: dict[str, float]
    supuestos: dict[str, float] = field(default_factory=dict)
    factor_demanda: float = 1.0


def construir_parametros(
    df_forecasts: pd.DataFrame,
    df_clean: pd.DataFrame,
    supuestos: dict[str, float] | None = None,
    factor_demanda: float = 1.0,
    cap_nodo: dict[str, float] | None = None,
) -> ParametrosRed:
    """Construye todos los parámetros del modelo a partir de los pronósticos y la historia."""
    s = {**SUPUESTOS_DEFAULT, **(supuestos or {})}
    cap_nodo = {**CAP_NODO_DEFAULT, **(cap_nodo or {})}

    # --- Demanda pronosticada D_pc (uds/mes) ---
    piv = df_forecasts.set_index(["Producto", "Cliente"])["Demanda_Pronosticada_Nov2026"]
    D = {
        (p, c): float(round(piv.get((p, c), 0.0) * factor_demanda))
        for p in PRODUCTOS_ORDENADOS
        for c in CLIENTES_ORDENADOS
    }

    # --- Precio de referencia por producto (mediana de las medianas por cliente) ---
    precios = (
        df_forecasts.groupby("Producto")["Precio_Mediano"].median().reindex(PRODUCTOS_ORDENADOS).astype(float).to_dict()
    )

    # --- Capacidad de producción Cap_fp: pico histórico mensual del producto ---
    hist = df_clean.copy()
    hist["Fecha"] = pd.to_datetime(hist["Fecha"])
    pico = hist.groupby(["Fecha", "Producto"])["Demanda"].sum().groupby("Producto").max()
    Cap, CP = {}, {}
    for p in PRODUCTOS_ORDENADOS:
        cap_principal = math.ceil(float(pico.get(p, 0.0)) * (1 + s["margen_capacidad"]) / 10) * 10
        base_cp = s["pct_costo_produccion"] * precios[p]
        for f in PLANTAS_OFICIALES:
            es_asignada = MAPA_PRODUCTO_PLANTA.get(p) == f
            Cap[(f, p)] = float(cap_principal if es_asignada else round(cap_principal * s["pct_capacidad_secundaria"]))
            recargo = 0.0 if es_asignada else s["recargo_planta_secundaria"]
            CP[(f, p)] = round(base_cp * FACTOR_COSTO_PLANTA[f] * (1 + recargo), 2)

    # --- Costos de transporte y distribución (COP/ud) ---
    CT, CD = {}, {}
    for p in PRODUCTOS_ORDENADOS:
        seguro = s["pct_seguro_carga"] * precios[p]
        for f in PLANTAS_OFICIALES:
            for n in NODOS:
                CT[(f, n, p)] = round(s["tarifa_troncal"] * DIST_PLANTA_NODO[(f, n)] + seguro, 2)
        for n in NODOS:
            for c in CLIENTES_ORDENADOS:
                CD[(n, c, p)] = round(
                    s["tarifa_distribucion"] * DIST_NODO_CLIENTE[(n, c)] + seguro + s["costo_cross_docking"], 2
                )

    # --- Capacidades de transporte (uds/mes) ---
    CapT = {k: float(v * s["uds_por_camion"]) for k, v in CAMIONES_PLANTA_NODO.items()}
    CapD = {k: float(v * s["uds_por_camion"]) for k, v in CAMIONES_NODO_CLIENTE.items()}

    # --- Penalización por unidad no atendida (solo si se permiten faltantes) ---
    Penal = {(p, c): round(s["factor_penalizacion"] * precios[p], 2) for p in PRODUCTOS_ORDENADOS for c in CLIENTES_ORDENADOS}

    return ParametrosRed(
        D=D, Cap=Cap, CP=CP, CT=CT, CD=CD, CapN={n: float(cap_nodo[n]) for n in NODOS},
        CapT=CapT, CapD=CapD, Penal=Penal, precios=precios, supuestos=s, factor_demanda=factor_demanda,
    )


# ==============================================================================
# 3. Modelo de optimización con PuLP
# ==============================================================================
def _idx(lista: list[str], x: str) -> int:
    return lista.index(x) + 1


def resolver_modelo(par: ParametrosRed, permitir_faltantes: bool = False) -> dict:
    """Formula y resuelve el modelo de costo mínimo. Devuelve estado, costos y tablas de decisión."""
    P, C, F, N = PRODUCTOS_ORDENADOS, CLIENTES_ORDENADOS, PLANTAS_OFICIALES, NODOS
    prob = pulp.LpProblem("Plan_Produccion_Trasbordo_Distribucion", pulp.LpMinimize)

    # Variables (nombres ASCII para evitar problemas de codificación con el solver)
    Q = {(f, p): prob.add_variable(f"Q_f{_idx(F, f)}_p{_idx(P, p)}", lowBound=0) for f in F for p in P}
    X = {
        (f, n, p): prob.add_variable(f"X_f{_idx(F, f)}_n{_idx(N, n)}_p{_idx(P, p)}", lowBound=0)
        for f in F for n in N for p in P
    }
    Y = {
        (n, c, p): prob.add_variable(f"Y_n{_idx(N, n)}_c{_idx(C, c)}_p{_idx(P, p)}", lowBound=0)
        for n in N for c in C for p in P
    }
    S = (
        {(p, c): prob.add_variable(f"S_p{_idx(P, p)}_c{_idx(C, c)}", lowBound=0) for p in P for c in C}
        if permitir_faltantes
        else {}
    )

    # Función objetivo
    costo_prod = pulp.lpSum(par.CP[f, p] * Q[f, p] for f in F for p in P)
    costo_transp = pulp.lpSum(par.CT[f, n, p] * X[f, n, p] for f in F for n in N for p in P)
    costo_dist = pulp.lpSum(par.CD[n, c, p] * Y[n, c, p] for n in N for c in C for p in P)
    costo_penal = pulp.lpSum(par.Penal[p, c] * S[p, c] for p in P for c in C) if permitir_faltantes else 0
    prob += costo_prod + costo_transp + costo_dist + costo_penal, "Costo_Total"

    # R1. Satisfacción de la demanda
    for p in P:
        for c in C:
            atendido = pulp.lpSum(Y[n, c, p] for n in N)
            if permitir_faltantes:
                prob += atendido + S[p, c] == par.D[p, c], f"R1_Demanda_p{_idx(P, p)}_c{_idx(C, c)}"
            else:
                prob += atendido == par.D[p, c], f"R1_Demanda_p{_idx(P, p)}_c{_idx(C, c)}"
    # R2. Capacidad de producción
    for f in F:
        for p in P:
            prob += Q[f, p] <= par.Cap[f, p], f"R2_CapProd_f{_idx(F, f)}_p{_idx(P, p)}"
    # R3. Balance de flujo en plantas
    for f in F:
        for p in P:
            prob += Q[f, p] == pulp.lpSum(X[f, n, p] for n in N), f"R3_BalPlanta_f{_idx(F, f)}_p{_idx(P, p)}"
    # R4. Balance de flujo en nodos
    for n in N:
        for p in P:
            prob += (
                pulp.lpSum(X[f, n, p] for f in F) == pulp.lpSum(Y[n, c, p] for c in C),
                f"R4_BalNodo_n{_idx(N, n)}_p{_idx(P, p)}",
            )
    # R5. Capacidad de procesamiento de nodos
    for n in N:
        prob += pulp.lpSum(X[f, n, p] for f in F for p in P) <= par.CapN[n], f"R5_CapNodo_n{_idx(N, n)}"
    # R6. Capacidades de transporte (planta→nodo y nodo→cliente)
    for f in F:
        for n in N:
            prob += pulp.lpSum(X[f, n, p] for p in P) <= par.CapT[f, n], f"R6a_CapT_f{_idx(F, f)}_n{_idx(N, n)}"
    for n in N:
        for c in C:
            prob += pulp.lpSum(Y[n, c, p] for p in P) <= par.CapD[n, c], f"R6b_CapD_n{_idx(N, n)}_c{_idx(C, c)}"
    # R7. No negatividad: definida con lowBound=0 en todas las variables.

    # PuLP >= 4: add_variable, constraints() y LpSolveStats.status_str.
    # PuLP 3.3.x mantiene add_variable, pero devuelve un código y guarda prob.status.
    # `pulp[cbc]` instala el ejecutable CBC requerido por COIN_CMD en PuLP 4.x.
    contar_restricciones = lambda: len(prob.constraints() if callable(prob.constraints) else prob.constraints)
    try:
        solver = pulp.COIN_CMD(msg=False)
        if not solver.available():
            raise RuntimeError("CBC no está disponible. Instale `pulp[cbc]` o configure CBC en PATH.")
        estadisticas = prob.solve(solver)
        if hasattr(estadisticas, "status_str"):
            estado = estadisticas.status_str  # PuLP 4.x
        else:
            estado = pulp.LpStatus[prob.status]  # PuLP 3.x
    except Exception as exc:
        return {
            "estado": "Solver no disponible", "optimo": False,
            "permitir_faltantes": permitir_faltantes,
            "n_variables": len(prob.variables()),
            "n_restricciones": contar_restricciones(),
            "parametros": par,
            "diagnostico": [f"No se pudo ejecutar el solucionador de programación lineal: {exc}"],
        }

    resultado: dict = {
        "estado": estado,
        "optimo": estado == "Optimal",
        "permitir_faltantes": permitir_faltantes,
        "n_variables": len(prob.variables()),
        "n_restricciones": contar_restricciones(),
        "parametros": par,
    }
    if estado != "Optimal":
        resultado["diagnostico"] = diagnosticar_infactibilidad(par)
        return resultado

    v = lambda var: max(0.0, float(var.varValue or 0.0))  # noqa: E731

    df_Q = pd.DataFrame(
        [{"Planta": f, "Producto": p, "Cantidad": v(Q[f, p]), "Capacidad": par.Cap[f, p], "Costo_Unitario": par.CP[f, p]}
         for f in F for p in P]
    )
    df_Q["Costo_Total"] = df_Q["Cantidad"] * df_Q["Costo_Unitario"]
    df_Q["Utilizacion_%"] = (df_Q["Cantidad"] / df_Q["Capacidad"].where(df_Q["Capacidad"] > 0) * 100).fillna(0)

    df_X = pd.DataFrame(
        [{"Planta": f, "Nodo": n, "Producto": p, "Cantidad": v(X[f, n, p]), "Costo_Unitario": par.CT[f, n, p]}
         for f in F for n in N for p in P]
    )
    df_X["Costo_Total"] = df_X["Cantidad"] * df_X["Costo_Unitario"]

    df_Y = pd.DataFrame(
        [{"Nodo": n, "Cliente": c, "Producto": p, "Cantidad": v(Y[n, c, p]), "Costo_Unitario": par.CD[n, c, p]}
         for n in N for c in C for p in P]
    )
    df_Y["Costo_Total"] = df_Y["Cantidad"] * df_Y["Costo_Unitario"]

    df_S = pd.DataFrame(
        [{"Producto": p, "Cliente": c, "Demanda": par.D[p, c],
          "Faltante": v(S[p, c]) if permitir_faltantes else 0.0} for p in P for c in C]
    )
    df_S["Atendido"] = df_S["Demanda"] - df_S["Faltante"]

    c_prod = float(df_Q["Costo_Total"].sum())
    c_transp = float(df_X["Costo_Total"].sum())
    c_dist = float(df_Y["Costo_Total"].sum())
    c_penal = float(sum(par.Penal[p, c] * v(S[p, c]) for p in P for c in C)) if permitir_faltantes else 0.0

    verificacion = verificar_solucion(par, df_Q, df_X, df_Y, df_S)
    resultado.update(
        {
            "Z": float(pulp.value(prob.objective)),
            "costo_produccion": c_prod,
            "costo_transporte": c_transp,
            "costo_distribucion": c_dist,
            "costo_penalizacion": c_penal,
            "df_Q": df_Q,
            "df_X": df_X,
            "df_Y": df_Y,
            "df_S": df_S,
            "utilizacion": calcular_utilizaciones(par, df_Q, df_X, df_Y),
            "verificacion": verificacion,
        }
    )
    if not bool(verificacion["Cumple"].all()):
        resultado["optimo"] = False
        resultado["estado"] = "Verificación fallida"
        resultado["diagnostico"] = [
            "La solución numérica no cumple todas las restricciones dentro de la "
            "tolerancia establecida. No se presenta como óptima."
        ]
    return resultado


# ==============================================================================
# 4. Indicadores, verificación y diagnóstico
# ==============================================================================
def calcular_utilizaciones(par: ParametrosRed, df_Q, df_X, df_Y) -> dict[str, pd.DataFrame]:
    """Utilización de plantas, nodos y arcos de transporte."""
    planta = df_Q.groupby("Planta", sort=False)[["Cantidad", "Capacidad"]].sum().reindex(PLANTAS_OFICIALES).reset_index()
    planta["Utilizacion_%"] = planta["Cantidad"] / planta["Capacidad"] * 100

    nodo = df_X.groupby("Nodo")["Cantidad"].sum().reindex(NODOS).reset_index()
    nodo["Capacidad"] = nodo["Nodo"].map(par.CapN)
    nodo["Utilizacion_%"] = nodo["Cantidad"] / nodo["Capacidad"] * 100

    arcos_t = df_X.groupby(["Planta", "Nodo"])["Cantidad"].sum().reset_index()
    arcos_t["Capacidad"] = [par.CapT[(f, n)] for f, n in zip(arcos_t["Planta"], arcos_t["Nodo"])]
    arcos_t["Utilizacion_%"] = (arcos_t["Cantidad"] / arcos_t["Capacidad"].where(arcos_t["Capacidad"] > 0) * 100).fillna(0)
    arcos_t["Permitido"] = arcos_t["Capacidad"] > 0

    arcos_d = df_Y.groupby(["Nodo", "Cliente"])["Cantidad"].sum().reset_index()
    arcos_d["Capacidad"] = [par.CapD[(n, c)] for n, c in zip(arcos_d["Nodo"], arcos_d["Cliente"])]
    arcos_d["Utilizacion_%"] = (arcos_d["Cantidad"] / arcos_d["Capacidad"].where(arcos_d["Capacidad"] > 0) * 100).fillna(0)
    arcos_d["Permitido"] = arcos_d["Capacidad"] > 0

    return {"plantas": planta, "nodos": nodo, "arcos_planta_nodo": arcos_t, "arcos_nodo_cliente": arcos_d}


def verificar_solucion(par: ParametrosRed, df_Q, df_X, df_Y, df_S, tol: float = 1e-4) -> pd.DataFrame:
    """Comprueba de forma independiente que la solución cumple todas las restricciones."""
    chk = []
    atendido = df_Y.groupby(["Producto", "Cliente"])["Cantidad"].sum()
    falt = df_S.set_index(["Producto", "Cliente"])["Faltante"]
    err1 = max(abs(atendido.get((p, c), 0) + falt.get((p, c), 0) - par.D[p, c]) for p, c in par.D)
    chk.append(("R1 Satisfacción de la demanda", err1 <= tol, err1))

    err2 = float((df_Q["Cantidad"] - df_Q["Capacidad"]).max())
    chk.append(("R2 Capacidad de producción", err2 <= tol, max(err2, 0)))

    salida = df_X.groupby(["Planta", "Producto"])["Cantidad"].sum()
    err3 = float((df_Q.set_index(["Planta", "Producto"])["Cantidad"] - salida).abs().max())
    chk.append(("R3 Balance en plantas", err3 <= tol, err3))

    ent = df_X.groupby(["Nodo", "Producto"])["Cantidad"].sum()
    sal = df_Y.groupby(["Nodo", "Producto"])["Cantidad"].sum()
    err4 = float((ent - sal).abs().max())
    chk.append(("R4 Balance en nodos", err4 <= tol, err4))

    flujo_n = df_X.groupby("Nodo")["Cantidad"].sum()
    err5 = max(flujo_n.get(n, 0) - par.CapN[n] for n in NODOS)
    chk.append(("R5 Capacidad de nodos", err5 <= tol, max(err5, 0)))

    ft = df_X.groupby(["Planta", "Nodo"])["Cantidad"].sum()
    fd = df_Y.groupby(["Nodo", "Cliente"])["Cantidad"].sum()
    err6 = max(max(ft.get(k, 0) - cap for k, cap in par.CapT.items()), max(fd.get(k, 0) - cap for k, cap in par.CapD.items()))
    chk.append(("R6 Capacidades de transporte", err6 <= tol, max(err6, 0)))

    min_val = min(df_Q["Cantidad"].min(), df_X["Cantidad"].min(), df_Y["Cantidad"].min())
    chk.append(("R7 No negatividad", min_val >= -tol, abs(min(min_val, 0))))

    return pd.DataFrame(chk, columns=["Restricción", "Cumple", "Máxima desviación (uds)"])


def diagnosticar_infactibilidad(par: ParametrosRed) -> list[str]:
    """Identifica cuellos de botella evidentes cuando el modelo no es factible."""
    msgs = []
    for p in PRODUCTOS_ORDENADOS:
        dem = sum(par.D[p, c] for c in CLIENTES_ORDENADOS)
        cap = sum(par.Cap[f, p] for f in PLANTAS_OFICIALES)
        if dem > cap:
            msgs.append(f"Producción insuficiente de {p}: demanda {dem:,.0f} uds > capacidad total {cap:,.0f} uds.")
    dem_total = sum(par.D.values())
    if dem_total > sum(par.CapN.values()):
        msgs.append(f"Nodos insuficientes: demanda total {dem_total:,.0f} uds > capacidad conjunta de nodos {sum(par.CapN.values()):,.0f} uds.")
    for c in CLIENTES_ORDENADOS:
        dem_c = sum(par.D[p, c] for p in PRODUCTOS_ORDENADOS)
        cap_c = sum(par.CapD[n, c] for n in NODOS)
        if dem_c > cap_c:
            msgs.append(f"Distribución insuficiente hacia {c}: demanda {dem_c:,.0f} uds > capacidad de arcos {cap_c:,.0f} uds.")
    for f in PLANTAS_OFICIALES:
        cap_salida = sum(par.CapT[f, n] for n in NODOS)
        min_requerido = 0.0
        for p in PRODUCTOS_ORDENADOS:
            dem = sum(par.D[p, c] for c in CLIENTES_ORDENADOS)
            otras = sum(par.Cap[g, p] for g in PLANTAS_OFICIALES if g != f)
            min_requerido += max(0.0, dem - otras)
        if min_requerido > cap_salida:
            msgs.append(f"Transporte insuficiente desde {f}: debe despachar al menos {min_requerido:,.0f} uds y su flota permite {cap_salida:,.0f} uds.")
    if not msgs:
        msgs.append("Ninguna capacidad individual es menor que la demanda; la infactibilidad proviene de la combinación de restricciones. Active 'Permitir faltantes' para ver qué demanda queda sin atender.")
    return msgs


# ==============================================================================
# 5. Tablas de parámetros para mostrar en la app
# ==============================================================================
def tablas_parametros(par: ParametrosRed) -> dict[str, pd.DataFrame]:
    P, C, F, N = PRODUCTOS_ORDENADOS, CLIENTES_ORDENADOS, PLANTAS_OFICIALES, NODOS
    D = pd.DataFrame([[par.D[p, c] for c in C] for p in P], index=P, columns=C)
    D["Total"] = D.sum(axis=1)
    Cap = pd.DataFrame([[par.Cap[f, p] for f in F] for p in P], index=P, columns=F)
    Cap["Planta asignada"] = [MAPA_PRODUCTO_PLANTA[p] for p in P]
    CP = pd.DataFrame([[par.CP[f, p] for f in F] for p in P], index=P, columns=F)
    CT = pd.DataFrame(
        [[par.CT[f, n, p] for f in F for n in N] for p in P], index=P,
        columns=[f"{f} → {n}" for f in F for n in N],
    )
    CD = pd.DataFrame(
        [[par.CD[n, c, p] for n in N for c in C] for p in P], index=P,
        columns=[f"{n} → {c}" for n in N for c in C],
    )
    CapN = pd.DataFrame(
        {"Nodo": N, "Ubicación": [UBICACION_NODOS[n] for n in N], "CapN (uds/mes)": [par.CapN[n] for n in N]}
    )
    CapT = pd.DataFrame(
        [{"Planta": f, "Nodo": n, "Distancia (km)": DIST_PLANTA_NODO[f, n],
          "Camiones/mes": CAMIONES_PLANTA_NODO[f, n], "CapT (uds/mes)": par.CapT[f, n],
          "Permitido": par.CapT[f, n] > 0} for f in F for n in N]
    )
    CapD = pd.DataFrame(
        [{"Nodo": n, "Cliente": c, "Ubicación cliente": UBICACION_CLIENTES[c], "Distancia (km)": DIST_NODO_CLIENTE[n, c],
          "Camiones/mes": CAMIONES_NODO_CLIENTE[n, c], "CapD (uds/mes)": par.CapD[n, c],
          "Permitido": par.CapD[n, c] > 0} for n in N for c in C]
    )
    return {"D": D, "Cap": Cap, "CP": CP, "CT": CT, "CD": CD, "CapN": CapN, "CapT": CapT, "CapD": CapD}


# ==============================================================================
# 6. Visualizaciones
# ==============================================================================
COLOR_PLANTA = {"Bogotá": "#0f4c81", "Medellín": "#1b7895", "Cali": "#f28e2b", "Bucaramanga": "#59a14f"}


def plot_sankey_red(res: dict) -> go.Figure:
    """Diagrama de flujo Plantas → Nodos → Clientes con las cantidades óptimas."""
    etiquetas = PLANTAS_OFICIALES + NODOS + [f"{c} ({UBICACION_CLIENTES[c]})" for c in CLIENTES_ORDENADOS]
    idx = {k: i for i, k in enumerate(PLANTAS_OFICIALES + NODOS + CLIENTES_ORDENADOS)}
    fx = res["df_X"].groupby(["Planta", "Nodo"])["Cantidad"].sum().reset_index()
    fy = res["df_Y"].groupby(["Nodo", "Cliente"])["Cantidad"].sum().reset_index()
    fx, fy = fx[fx["Cantidad"] > 0.5], fy[fy["Cantidad"] > 0.5]
    src = [idx[a] for a in fx["Planta"]] + [idx[a] for a in fy["Nodo"]]
    tgt = [idx[b] for b in fx["Nodo"]] + [idx[b] for b in fy["Cliente"]]
    val = list(fx["Cantidad"]) + list(fy["Cantidad"])
    colores_nodo = [COLOR_PLANTA[f] for f in PLANTAS_OFICIALES] + ["#6b7280", "#374151"] + ["#94a3b8"] * len(CLIENTES_ORDENADOS)
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=etiquetas, pad=18, thickness=18, color=colores_nodo),
            link=dict(source=src, target=tgt, value=val, color="rgba(15,76,129,0.25)"),
            valueformat=",.0f",
            valuesuffix=" uds",
        )
    )
    fig.update_layout(title="Flujo óptimo: Plantas → Nodos de trasbordo → Clientes (uds/mes)", height=480,
                      margin=dict(l=20, r=20, t=60, b=20), template="plotly_white")
    return fig


def plot_produccion_por_planta(df_Q: pd.DataFrame) -> go.Figure:
    data = df_Q[df_Q["Cantidad"] > 0.5]
    fig = px.bar(
        data, x="Producto", y="Cantidad", color="Planta", color_discrete_map=COLOR_PLANTA,
        category_orders={"Producto": PRODUCTOS_ORDENADOS, "Planta": PLANTAS_OFICIALES},
        text_auto=",.0f", labels={"Cantidad": "Unidades a producir (Q_fp)"},
    )
    fig.update_layout(title="¿Qué, cuánto y dónde producir? — Plan óptimo Q_fp", barmode="stack",
                      template="plotly_white", height=420, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_utilizacion(df: pd.DataFrame, etiqueta: str, titulo: str) -> go.Figure:
    colores = ["#d62728" if u >= 99.5 else "#f28e2b" if u >= 85 else "#0f4c81" for u in df["Utilizacion_%"]]
    fig = go.Figure(
        go.Bar(x=df[etiqueta], y=df["Utilizacion_%"], marker_color=colores,
               text=[f"{u:.1f}%" for u in df["Utilizacion_%"]], textposition="outside")
    )
    fig.add_hline(y=100, line_dash="dash", line_color="#d62728")
    fig.update_layout(title=titulo, yaxis_title="Utilización (%)", yaxis_range=[0, 115],
                      template="plotly_white", height=360, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_composicion_costos(res: dict) -> go.Figure:
    comp = pd.DataFrame(
        {
            "Componente": ["Producción", "Transporte planta→nodo", "Distribución nodo→cliente", "Penalización faltantes"],
            "Costo": [res["costo_produccion"], res["costo_transporte"], res["costo_distribucion"], res["costo_penalizacion"]],
        }
    )
    comp = comp[comp["Costo"] > 0]
    fig = px.pie(comp, names="Componente", values="Costo", hole=0.5,
                 color_discrete_sequence=["#0f4c81", "#1b7895", "#f28e2b", "#d62728"])
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(title="Composición del costo total mínimo Z", template="plotly_white",
                      height=380, margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    return fig


if __name__ == "__main__":
    fc = pd.read_csv("data/pronosticos_nov2026.csv")
    hc = pd.read_csv("data/demanda_limpia_1800.csv")
    r = resolver_modelo(construir_parametros(fc, hc))
    print(r["estado"], f"Z = ${r.get('Z', float('nan')):,.0f} COP")
