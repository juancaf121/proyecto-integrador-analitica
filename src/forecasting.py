"""
Módulo de Pronósticos de Demanda (Fases 3 y 4) para la Plataforma Inteligente
de Planeación de la Producción y Distribución.

Implementa 4 metodologías de pronóstico sobre las 50 series individuales
Producto × Cliente (36 meses históricos):
1. Promedio Móvil simple (ventana k configurable)
2. Suavizamiento Exponencial Doble de Holt
3. Suavizamiento Exponencial Triple de Holt-Winters (period=12, aditivo/multiplicativo)
4. Red Neuronal Multicapa (MLPRegressor con ventana deslizante y StandardScaler)

Aplica validación temporal estricta (walk-forward 1 paso adelante sobre los
últimos 6 meses), selecciona automáticamente el modelo ganador por serie
minimizando MAPE/RMSE/MAE y genera el pronóstico de demanda D_{p,c} para
Noviembre de 2026 (t = 37).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.etl import (
    CLIENTES_ORDENADOS,
    MAPA_PRODUCTO_PLANTA,
    PRODUCTOS_ORDENADOS,
    load_clean_data,
)

MODEL_NAMES: list[str] = ["Promedio móvil", "Holt", "Holt-Winters", "MLP"]

MODEL_COLORS: dict[str, str] = {
    "Promedio móvil": "#6c757d",
    "Holt": "#ff7f0e",
    "Holt-Winters": "#2ca02c",
    "MLP": "#9467bd",
}

FORECAST_DATE_STR = "2026-11-01"


# ==============================================================================
# Funciones individuales de los 4 modelos (1 paso adelante)
# ==============================================================================
def forecast_moving_average(hist: np.ndarray, window: int = 3) -> float:
    """Promedio aritmético simple de las últimas `window` observaciones."""
    k = max(1, min(int(window), len(hist)))
    return float(max(0.0, np.mean(hist[-k:])))


def forecast_holt(hist: np.ndarray, fallback_window: int = 3) -> float:
    """Suavizamiento exponencial doble con tendencia aditiva (sin estacionalidad)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                hist.astype(float),
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            pred = float(model.forecast(1)[0])
        if np.isnan(pred) or np.isinf(pred):
            return forecast_moving_average(hist, fallback_window)
        return float(max(0.0, pred))
    except Exception:
        return forecast_moving_average(hist, fallback_window)


def forecast_holt_winters(
    hist: np.ndarray,
    seasonal: str = "add",
    seasonal_periods: int = 12,
    fallback_window: int = 3,
) -> float:
    """Suavizamiento exponencial triple de Holt-Winters (nivel, tendencia y ciclo anual)."""
    try:
        arr = hist.astype(float)
        if seasonal == "mul" and np.any(arr <= 0):
            raise ValueError("Holt-Winters multiplicativo requiere valores estrictamente positivos.")
        if len(arr) < 2 * seasonal_periods:
            return forecast_holt(arr, fallback_window)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                arr,
                trend="add",
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True)
            pred = float(model.forecast(1)[0])
        if np.isnan(pred) or np.isinf(pred):
            return forecast_holt(arr, fallback_window)
        return float(max(0.0, pred))
    except Exception:
        return forecast_holt(hist, fallback_window)


def forecast_mlp(
    hist: np.ndarray,
    window_lags: int = 4,
    max_iter: int = 200,
    random_state: int = 42,
) -> float:
    """
    Red Neuronal Multicapa (32, 16) con ReLU y Adam sobre ventanas deslizantes
    X_t = [D_{t-W}, ..., D_{t-1}] -> Y_t = D_t.
    Ajusta StandardScaler únicamente sobre la partición de entrenamiento.
    """
    arr = hist.astype(float)
    w = max(2, min(int(window_lags), len(arr) - 2))

    X = np.array([arr[i - w : i] for i in range(w, len(arr))], dtype=float)
    y = arr[w:].astype(float)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        mlp = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.01,
            max_iter=int(max_iter),
            random_state=random_state,
        )
        mlp.fit(X_scaled, y_scaled)

    x_next = arr[-w:].reshape(1, -1)
    x_next_scaled = scaler_x.transform(x_next)
    y_next_scaled = mlp.predict(x_next_scaled)
    pred = float(scaler_y.inverse_transform(y_next_scaled.reshape(-1, 1))[0, 0])
    if np.isnan(pred) or np.isinf(pred):
        return forecast_moving_average(arr, w)
    return float(max(0.0, pred))


# ==============================================================================
# Métricas de evaluación
# ==============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE, RMSE y MAPE (%) sobre la ventana de validación."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(yt - yp)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mape = float(100.0 * np.mean(abs_err / np.maximum(yt, 1.0)))
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
    }


# ==============================================================================
# Motor principal: Walk-forward de las 50 series y selección automática
# ==============================================================================
def _predecir_un_paso(
    model_name: str,
    hist: np.ndarray,
    ma_window: int,
    hw_seasonal: str,
    mlp_lags: int,
    mlp_max_iter: int,
) -> float:
    if model_name == "Promedio móvil":
        return forecast_moving_average(hist, window=ma_window)
    if model_name == "Holt":
        return forecast_holt(hist, fallback_window=ma_window)
    if model_name == "Holt-Winters":
        return forecast_holt_winters(
            hist,
            seasonal=hw_seasonal,
            seasonal_periods=12,
            fallback_window=ma_window,
        )
    if model_name == "MLP":
        return forecast_mlp(hist, window_lags=mlp_lags, max_iter=mlp_max_iter)
    raise ValueError(f"Modelo desconocido: {model_name}")


def run_forecasting_pipeline(
    df_clean: Optional[pd.DataFrame] = None,
    clean_csv_path: Union[str, Path] = "data/demanda_limpia_1800.csv",
    ma_window: int = 3,
    hw_seasonal: str = "add",
    mlp_lags: int = 4,
    mlp_max_iter: int = 200,
    criterio: str = "MAPE",
    output_forecasts_path: Union[str, Path, None] = "data/pronosticos_nov2026.csv",
    output_metrics_path: Union[str, Path, None] = "data/metricas_modelos_50series.csv",
) -> dict[str, Any]:
    """
    Ejecuta el entrenamiento y validación walk-forward (30 train / 6 test) para
    las 50 series Producto × Cliente, selecciona el modelo ganador según `criterio`
    ('MAPE', 'RMSE' o 'MAE') y genera los pronósticos finales para Nov-2026 (t=37).
    """
    if df_clean is None:
        df = load_clean_data(clean_csv_path)
    else:
        df = df_clean.copy()
        df["Fecha"] = pd.to_datetime(df["Fecha"])

    if criterio not in ("MAPE", "RMSE", "MAE"):
        raise ValueError("`criterio` debe ser 'MAPE', 'RMSE' o 'MAE'.")

    metrics_rows: list[dict[str, Any]] = []
    forecast_rows: list[dict[str, Any]] = []
    series_details: dict[tuple[str, str], dict[str, Any]] = {}

    train_len = 30
    test_len = 6

    for prod in PRODUCTOS_ORDENADOS:
        for cli in CLIENTES_ORDENADOS:
            sub = df[(df["Producto"] == prod) & (df["Cliente"] == cli)].sort_values("Fecha")
            if len(sub) != train_len + test_len:
                raise ValueError(
                    f"La serie {prod} - {cli} tiene {len(sub)} períodos (se esperaban 36)."
                )

            fechas = sub["Fecha"].values
            y_all = sub["Demanda"].values.astype(float)
            y_val_true = y_all[train_len:]
            planta = MAPA_PRODUCTO_PLANTA[prod]
            precio_med = float(sub["Precio"].median())

            val_preds: dict[str, np.ndarray] = {}
            nov_preds: dict[str, int] = {}
            model_metrics: dict[str, dict[str, float]] = {}

            for model_name in MODEL_NAMES:
                preds_6 = []
                for step in range(test_len):
                    hist_step = y_all[: train_len + step]
                    pred_step = _predecir_un_paso(
                        model_name,
                        hist_step,
                        ma_window=ma_window,
                        hw_seasonal=hw_seasonal,
                        mlp_lags=mlp_lags,
                        mlp_max_iter=mlp_max_iter,
                    )
                    preds_6.append(pred_step)

                arr_preds_6 = np.array(preds_6, dtype=float)
                val_preds[model_name] = arr_preds_6
                model_metrics[model_name] = compute_metrics(y_val_true, arr_preds_6)

                pred_nov = _predecir_un_paso(
                    model_name,
                    y_all,
                    ma_window=ma_window,
                    hw_seasonal=hw_seasonal,
                    mlp_lags=mlp_lags,
                    mlp_max_iter=mlp_max_iter,
                )
                nov_preds[model_name] = int(round(max(0.0, pred_nov)))

            ganador = min(MODEL_NAMES, key=lambda m: model_metrics[m][criterio])
            demanda_ganadora = nov_preds[ganador]
            ingreso_pron = round(demanda_ganadora * precio_med, 2)

            metrics_rows.append(
                {
                    "Producto": prod,
                    "Cliente": cli,
                    "Planta": planta,
                    "Precio_Mediano": round(precio_med, 2),
                    "MAE_Promedio_movil": model_metrics["Promedio móvil"]["MAE"],
                    "RMSE_Promedio_movil": model_metrics["Promedio móvil"]["RMSE"],
                    "MAPE_Promedio_movil": model_metrics["Promedio móvil"]["MAPE"],
                    "Pred_Nov2026_Promedio_movil": nov_preds["Promedio móvil"],
                    "MAE_Holt": model_metrics["Holt"]["MAE"],
                    "RMSE_Holt": model_metrics["Holt"]["RMSE"],
                    "MAPE_Holt": model_metrics["Holt"]["MAPE"],
                    "Pred_Nov2026_Holt": nov_preds["Holt"],
                    "MAE_Holt_Winters": model_metrics["Holt-Winters"]["MAE"],
                    "RMSE_Holt_Winters": model_metrics["Holt-Winters"]["RMSE"],
                    "MAPE_Holt_Winters": model_metrics["Holt-Winters"]["MAPE"],
                    "Pred_Nov2026_Holt_Winters": nov_preds["Holt-Winters"],
                    "MAE_MLP": model_metrics["MLP"]["MAE"],
                    "RMSE_MLP": model_metrics["MLP"]["RMSE"],
                    "MAPE_MLP": model_metrics["MLP"]["MAPE"],
                    "Pred_Nov2026_MLP": nov_preds["MLP"],
                    "Criterio_Seleccion": criterio,
                    "Modelo_Ganador": ganador,
                    "MAE_Ganador": model_metrics[ganador]["MAE"],
                    "RMSE_Ganador": model_metrics[ganador]["RMSE"],
                    "MAPE_Ganador": model_metrics[ganador]["MAPE"],
                    "Demanda_Pronosticada_Nov2026": demanda_ganadora,
                    "Ingreso_Pronosticado_Nov2026": ingreso_pron,
                }
            )

            forecast_rows.append(
                {
                    "Fecha": FORECAST_DATE_STR,
                    "Producto": prod,
                    "Cliente": cli,
                    "Planta": planta,
                    "Modelo_Ganador": ganador,
                    "Demanda_Pronosticada_Nov2026": demanda_ganadora,
                    "MAE_Ganador": model_metrics[ganador]["MAE"],
                    "RMSE_Ganador": model_metrics[ganador]["RMSE"],
                    "MAPE_Ganador": model_metrics[ganador]["MAPE"],
                    "Precio_Mediano": round(precio_med, 2),
                    "Ingreso_Pronosticado_Nov2026": ingreso_pron,
                }
            )

            series_details[(prod, cli)] = {
                "fechas": pd.to_datetime(fechas),
                "y_real": y_all,
                "val_fechas": pd.to_datetime(fechas[train_len:]),
                "val_preds": val_preds,
                "nov_preds": nov_preds,
                "metrics": model_metrics,
                "winner": ganador,
                "planta": planta,
            }

    df_metrics = pd.DataFrame(metrics_rows)
    df_forecasts = pd.DataFrame(forecast_rows)
    df_product_summary = build_product_summary_table(df_metrics, criterio=criterio)

    if output_forecasts_path is not None:
        out_f = Path(output_forecasts_path)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        df_forecasts.to_csv(out_f, index=False, encoding="utf-8")

    if output_metrics_path is not None:
        out_m = Path(output_metrics_path)
        out_m.parent.mkdir(parents=True, exist_ok=True)
        df_metrics.to_csv(out_m, index=False, encoding="utf-8")

    return {
        "df_metrics": df_metrics,
        "df_forecasts": df_forecasts,
        "df_product_summary": df_product_summary,
        "series_details": series_details,
        "params": {
            "ma_window": ma_window,
            "hw_seasonal": hw_seasonal,
            "mlp_lags": mlp_lags,
            "mlp_max_iter": mlp_max_iter,
            "criterio": criterio,
        },
    }


def build_product_summary_table(
    df_metrics: pd.DataFrame,
    criterio: str = "MAPE",
) -> pd.DataFrame:
    """
    Construye la tabla resumen oficial por Producto:
    Producto | Promedio móvil | Holt | Holt-Winters | MLP | Ganador | Demanda Total Nov-2026
    promediando la métrica elegida entre los 5 clientes de cada producto.
    """
    col_map = {
        "Promedio móvil": f"{criterio}_Promedio_movil",
        "Holt": f"{criterio}_Holt",
        "Holt-Winters": f"{criterio}_Holt_Winters",
        "MLP": f"{criterio}_MLP",
    }

    rows = []
    for prod in PRODUCTOS_ORDENADOS:
        sub = df_metrics[df_metrics["Producto"] == prod]
        if sub.empty:
            continue
        promedios = {m: round(float(sub[col].mean()), 2) for m, col in col_map.items()}
        ganador_prod = min(MODEL_NAMES, key=lambda m: promedios[m])
        rows.append(
            {
                "Producto": prod,
                "Planta": MAPA_PRODUCTO_PLANTA[prod],
                "Promedio móvil": promedios["Promedio móvil"],
                "Holt": promedios["Holt"],
                "Holt-Winters": promedios["Holt-Winters"],
                "MLP": promedios["MLP"],
                "Ganador": ganador_prod,
                "Demanda_Total_Nov2026": int(sub["Demanda_Pronosticada_Nov2026"].sum()),
            }
        )
    return pd.DataFrame(rows)


# ==============================================================================
# Gráficos interactivos en Plotly (Fases 3 y 4)
# ==============================================================================
def plot_series_forecast_comparison(
    series_detail: dict[str, Any],
    producto: str,
    cliente: str,
) -> go.Figure:
    """
    Gráfico principal de la Fase 3:
    Serie histórica (36 meses) + Zona sombreada de validación (últimos 6 meses) +
    Ajuste de los 4 modelos en validación + Marcador destacado del pronóstico Nov-2026.
    """
    fechas = series_detail["fechas"]
    y_real = series_detail["y_real"]
    val_fechas = series_detail["val_fechas"]
    val_preds = series_detail["val_preds"]
    nov_preds = series_detail["nov_preds"]
    winner = series_detail["winner"]

    fig = go.Figure()

    # Zona sombreada de validación temporal (6 meses)
    fig.add_vrect(
        x0=val_fechas[0] - pd.Timedelta(days=12),
        x1=pd.Timestamp(FORECAST_DATE_STR) + pd.Timedelta(days=12),
        fillcolor="rgba(27, 120, 149, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Validación Walk-Forward (6m) + Nov-2026",
        annotation_position="top left",
    )

    # Serie real histórica
    fig.add_trace(
        go.Scatter(
            x=fechas,
            y=y_real,
            mode="lines+markers",
            name="Demanda Real (36 meses)",
            line=dict(color="#0f4c81", width=2.8),
            marker=dict(size=6),
            hovertemplate="<b>%{x|%Y-%m}</b><br>Real: %{y:,.0f} uds<extra></extra>",
        )
    )

    # Predicciones de validación de cada modelo
    for model_name in MODEL_NAMES:
        fig.add_trace(
            go.Scatter(
                x=val_fechas,
                y=val_preds[model_name],
                mode="lines+markers",
                name=f"{model_name} (Validación)",
                line=dict(color=MODEL_COLORS[model_name], width=2.0, dash="dash"),
                marker=dict(size=6),
                hovertemplate=f"<b>{model_name}</b> (%{{x|%Y-%m}})<br>Pred: %{{y:,.1f}} uds<extra></extra>",
            )
        )

    # Marcador destacado del pronóstico ganador en Noviembre 2026
    fecha_nov = pd.Timestamp(FORECAST_DATE_STR)
    pred_nov_win = nov_preds[winner]
    fig.add_trace(
        go.Scatter(
            x=[fechas[-1], fecha_nov],
            y=[y_real[-1], pred_nov_win],
            mode="lines+markers+text",
            name=f"Pronóstico Nov-2026 ({winner})",
            line=dict(color="#d62728", width=2.5, dash="dot"),
            marker=dict(size=[6, 13], color=["#0f4c81", "#d62728"], symbol=["circle", "star"]),
            text=[None, f"{pred_nov_win:,} uds"],
            textposition="top center",
            textfont=dict(color="#d62728", size=12),
            hovertemplate=f"<b>Pronóstico Nov-2026 ({winner})</b><br>Demanda: {pred_nov_win:,} uds<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Validación Walk-Forward y Pronóstico Nov-2026 — {producto} · {cliente} (Ganador: {winner})",
        template="plotly_white",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=70, b=50),
    )
    fig.update_xaxes(title_text="Período (Mes)", tickformat="%b %Y")
    fig.update_yaxes(title_text="Unidades Demandadas", tickformat=",.0f")
    return fig


def plot_winner_distribution_donut(df_metrics: pd.DataFrame) -> go.Figure:
    """Gráfico Donut con la distribución de victorias entre las 50 series."""
    counts = (
        df_metrics["Modelo_Ganador"]
        .value_counts()
        .reindex(MODEL_NAMES, fill_value=0)
        .reset_index()
    )
    counts.columns = ["Modelo", "Series_Ganadas"]
    counts = counts[counts["Series_Ganadas"] > 0]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["Modelo"],
                values=counts["Series_Ganadas"],
                hole=0.52,
                marker=dict(colors=[MODEL_COLORS.get(m, "#0f4c81") for m in counts["Modelo"]]),
                textinfo="label+value+percent",
                textposition="outside",
                hovertemplate="<b>%{label}</b><br>Series ganadas: %{value} de 50 (%{percent})<extra></extra>",
            )
        ]
    )
    fig.add_annotation(
        text="<b>50</b><br>Series",
        x=0.5,
        y=0.5,
        font_size=15,
        showarrow=False,
    )
    fig.update_layout(
        title="Distribución de Victorias por Metodología (50 Series)",
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=60, b=30),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


def plot_demand_matrix_heatmap(df_forecasts: pd.DataFrame) -> go.Figure:
    """Heatmap 10 × 5 de la matriz de demanda pronosticada D_{p,c} para Nov-2026."""
    piv = (
        df_forecasts.pivot(
            index="Producto",
            columns="Cliente",
            values="Demanda_Pronosticada_Nov2026",
        )
        .loc[PRODUCTOS_ORDENADOS, CLIENTES_ORDENADOS]
    )
    fig = px.imshow(
        piv,
        text_auto=",.0f",
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels=dict(x="Cliente", y="Producto", color="Demanda D_{p,c}"),
    )
    fig.update_layout(
        title="Matriz de Demanda Pronosticada D_{p,c} — Noviembre 2026 (10 × 5)",
        template="plotly_white",
        height=450,
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig


if __name__ == "__main__":
    res = run_forecasting_pipeline()
    print(
        f"Pronósticos generados: {len(res['df_forecasts'])} series | "
        f"Demanda Total Nov-2026: {res['df_forecasts']['Demanda_Pronosticada_Nov2026'].sum():,} uds | "
        f"MAPE promedio ganadores: {res['df_metrics']['MAPE_Ganador'].mean():.2f}%"
    )
