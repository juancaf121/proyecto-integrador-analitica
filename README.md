# Proyecto integrador — Analítica de Datos | Entregable 5

Aplicación interactiva y analítica para el estudio de accidentalidad vial y costos de atención hospitalaria en Bucaramanga (2018–2021), desarrollada en Streamlit.

> **Datos para demostración:** el equipo responsable indica que los dos CSV incluidos son muestras ficticias para fines académicos. Esta afirmación no se verificó de forma independiente. Antes de usar un repositorio público, se debe comprobar que no contengan información real identificable ni datos restringidos. Las métricas de ejemplo no deben presentarse como rendimiento validado en pacientes reales.

---

## 1. Origen y estructura de los datos

- `data/df_modelado.csv`: 28.276 filas y 22 columnas. Base de datos depurada utilizada para el ajuste y evaluación de modelos supervisados (Entregables 3 y 4).
- `data/df_segmentado_clusters.csv`: 28.276 filas y 25 columnas. Incorpora las columnas `PC1`, `PC2` y `cluster` correspondientes a la segmentación descriptiva original del Entregable 3.
- `fase2_experimento_comparativo_completo.csv`: tabla de verificación experimental que compara el desempeño de los clasificadores con y sin variables diagnósticas (`patologia_principal` y `region_principal`).

> **Nota sobre el Entregable 2:** El entregable exploratorio previo utilizaba una base preliminar de 28.346 registros. La presente aplicación toma como referencia única el dataset de modelado (28.276 registros) para asegurar consistencia con las etapas de reducción dimensional y modelado predictivo.

---

## 2. Reproducción local

### Requisitos del sistema
- **Versión de Python recomendada:** Python 3.11 (documentada en `.python-version`; seleccionar también Python 3.11 en las opciones avanzadas del despliegue si el proveedor no toma este archivo automáticamente).
- **Semilla global:** `RANDOM_STATE = 42`.
- **Partición analítica:** 80 % entrenamiento / 20 % prueba (estratificada por la variable `politraumatismo`).

### Instrucciones paso a paso
1. Abrir una terminal en el directorio raíz del proyecto.
2. Crear un entorno virtual aislado:
   ```bash
   python -m venv .venv
   ```
3. Activar el entorno virtual:
   - En Windows (PowerShell):
     ```powershell
     .venv\Scripts\activate
     ```
   - En Linux / macOS:
     ```bash
     source .venv/bin/activate
     ```
4. Instalar las dependencias estrictamente requeridas:
   ```bash
   pip install -r requirements.txt
   ```
5. Iniciar la aplicación Streamlit:
   ```bash
   streamlit run app.py
   ```
6. Abrir la dirección URL local provista en el navegador (habitualmente `http://localhost:8501`).

---

## 3. Dependencias del proyecto (`requirements.txt`)

El archivo `requirements.txt` contiene únicamente los paquetes necesarios para la ejecución del aplicativo en Python 3.11:

```text
streamlit>=1.38,<2
pandas>=2.0,<3
numpy>=1.24,<3
scikit-learn>=1.4,<2
plotly>=5.19,<7
```

---

## 4. Estructura de archivos del repositorio

```text
├── app.py                                       # Aplicación principal de Streamlit (7 módulos integrados)
├── modelo_cuaderno4.py                          # Transformador ExtractorPCAKMeans ajustado solo en train
├── requirements.txt                             # Especificación de paquetes instalables
├── .python-version                              # Fijación de Python 3.11 para Streamlit Cloud
├── .gitignore                                   # Exclusiones generales; permite los dos CSV de demostración
├── README.md                                    # Documentación técnica y metodológica del entregable
├── fase2_experimento_comparativo_completo.csv   # Resultados del experimento de sensibilidad de variables
├── data/
│   ├── df_modelado.csv                          # Dataset de modelado (conservado intacto)
│   └── df_segmentado_clusters.csv               # Dataset con clústeres del Entregable 3 (conservado intacto)
```

---

## 5. Metodología y rigor analítico

1. **Preprocesamiento sin fuga de información (*Data Leakage*):**
   - El escalado (`StandardScaler`), el análisis de componentes principales (`PCA`) y el agrupamiento (`K-Means`) se ajustan **exclusivamente sobre el 80 % de entrenamiento** a través del transformador `ExtractorPCAKMeans`.
   - Las componentes exportadas `PC1`, `PC2` y la etiqueta `cluster` del Entregable 3 son puramente descriptivas y **no se utilizan como predictores** supervisados de costos ni de politraumatismo.
2. **Evaluación de modelos de clasificación:**
   - La métrica de resumen es **Average Precision (AP)**, calculada mediante `average_precision_score`. AP y el área trapezoidal (`auc(recall, precision)`) utilizan reglas de integración diferentes y no deben comunicarse como si fueran equivalentes.
   - El umbral de corte estándar de **0,50** se utiliza con fines comparativos algorítmicos. **No representa un umbral clínico validado**, pues en la práctica médica los costos de Falsos Negativos y Falsos Positivos son asimétricos y requieren calibración formal.
3. **Regresión de costos hospitalarios:**
   - Se modela la variable continua en escala logarítmica $\log_{10}(\text{costos})$ para estabilizar la varianza ante valores atípicos multimillonarios.
   - La retransformación a pesos colombianos (COP) incorpora la corrección de sesgo no lineal de **Duan (*smearing factor*)**, estimada estrictamente sobre el conjunto de entrenamiento.
   - Un $R^2$ de **0,0611 en PCA 80 %** o **0,1744 en PCA completo**, ambos calculados en la escala $\log_{10}(\text{costos})$, refleja una **limitación predictiva importante** debida a la falta de variables clínicas intrahospitalarias determinantes (días de UCI, cirugías, insumos médicos), por lo que el modelo es exploratorio y no apto para tarificación individual ni presupuestación clínica.
4. **Verificación de fuga temporal y análisis de disponibilidad:**
   - El retiro de `patologia_principal` y `region_principal` produce una disminución de ROC-AUC de $0,96$ a $0,62$ y de AP de $0,70$ a $0,14$.
   - Se documenta que la caída en métricas **no demuestra matemáticamente una fuga por sí sola**; lo que la sustenta es la verificación del flujo operativo: dichas variables corresponden a la codificación diagnóstica (CIE-10) del egreso hospitalario y **no están disponibles al ingreso o en triaje**.
5. **Validación temporal:**
   - Al evaluar temporalmente el modelo (entrenando con datos $\le 2020$ y probando sobre 2021), el desempeño cae de forma notable. Este fenómeno responde a múltiples factores concurrentes: inflación médica en tarifas SOAT, cambios en la severidad de siniestros, variaciones en los protocolos administrativos de registro y las restricciones de movilidad por la pandemia COVID-19 como uno de los factores contextuales adicionales.

---

## 6. Alcance de los datos y publicación

- Para la entrega académica, el equipo identifica los archivos `data/df_modelado.csv` y `data/df_segmentado_clusters.csv` como **muestras ficticias de demostración**. No se comprobó de manera independiente su procedencia.
- El repositorio de esta versión **sí incluye deliberadamente ambos archivos**, de modo que Streamlit Community Cloud pueda cargar los datos y mostrar la aplicación de forma automática.
- El `.gitignore` evita añadir archivos adicionales por accidente, pero **no excluye** los dos CSV necesarios.
- Si en algún momento se sustituyen por registros hospitalarios reales, no deben publicarse sin autorización, evaluación de privacidad y las medidas de protección correspondientes.
- El análisis y las métricas del trabajo sirven para ilustrar el flujo metodológico del proyecto; no equivalen a validación clínica ni representan necesariamente resultados sobre pacientes reales.

---

## 7. Verificación independiente de regresión y escalas (semilla 42)

Resultados recalculados directamente a partir de `df_modelado.csv` y `modelo_cuaderno4.py` con 22.620 registros de entrenamiento y 5.656 de prueba:

| Métrica calculada sobre prueba | PCA 80 % (12 componentes) | PCA completo (40 componentes) |
| --- | ---: | ---: |
| R² sobre log₁₀(costos) | 0,061131 | 0,174416 |
| MAE sobre log₁₀(costos) | 0,361895 | 0,335171 |
| RMSE sobre log₁₀(costos) | 0,502998 | 0,471677 |
| **R² sobre COP con corrección de Duan** | **0,014418** | **−0,171842** |
| MAE sobre COP con corrección de Duan | $802.045 | $752.486 |
| RMSE sobre COP con corrección de Duan | $1.606.208 | $1.751.418 |

Estas cifras dependen de las versiones de los paquetes y de la partición de prueba. La cifra **R² = 0,0768** mencionada en un informe previo del agente no se reproduce con el código y los datos suministrados. No debe comunicarse como resultado verificado. El R² puede tomar valores negativos en COP; significa que la suma de errores cuadráticos supera a la de la predicción constante con la media del conjunto evaluado. **No mezclar R² en COP con R² en escala logarítmica**.

## 8. Cómo publicar la versión con CSV de demostración

1. Descomprimir el ZIP de entrega, abrir la carpeta `Proyecto_Entregable5_Streamlit` y usar su **contenido** como raíz del repositorio de GitHub.
2. Confirmar que `app.py`, `requirements.txt` y `modelo_cuaderno4.py` queden en la raíz, y que ambos archivos CSV permanezcan dentro de `data/`.
3. Subir los archivos al repositorio. Si se utiliza Git, el `.gitignore` permite versionar únicamente los dos CSV esperados dentro de `data/`.
4. En Streamlit Community Cloud, seleccionar el repositorio, la rama `main` y `app.py` como archivo principal. Elegir Python 3.11 en las opciones avanzadas cuando esté disponible.
5. Verificar que los cinco indicadores de Inicio y los gráficos de clustering se carguen automáticamente. Para cargar un conjunto diferente, la aplicación también admite la carga manual.
6. Antes de hacer público el repositorio, confirmar la naturaleza ficticia y las condiciones de uso de los datos.

## 9. Limitaciones y observaciones

- PCA/K-means del Entregable 3 son descriptivos; no se reutilizan sus clústeres exportados como predictores para clasificación o regresión.
- Los experimentos de clasificación con variables diagnósticas pueden no representar predicción temprana. El momento de captura de cada campo requiere verificación documental.
- Los costos ilustrativos ponderados de la matriz de confusión son supuestos y **no cifras de costos de atención reales**.
- La ejecución de Streamlit debe comprobarse manualmente en el computador donde esté instalado; el análisis independiente aquí confirmó los cálculos de las regresiones y la compilación del código, pero no una prueba visual del servidor.


## 10. Verificación de requisitos visuales del Word

- **Carga y análisis:** la sección *Carga y calidad* presenta `df.head(10)`, `DataFrame.info()` completo, tipos, faltantes, duplicados y tabla de auditoría. *Análisis exploratorio* presenta `describe()`, histogramas, barras y matriz de correlación de Spearman visualizable con Plotly o `seaborn.heatmap()`.
- **Explicación de modelos:** cada algoritmo dispone de una explicación breve del funcionamiento y sus limitaciones. Los clasificadores presentan matriz de confusión, clasificación por clase, ROC-AUC y curva ROC y Precision–Recall.
- **PCA/K-means:** la sección descriptiva muestra varianza acumulada, variables con cargas dominantes y diagnóstico Elbow, Silhouette y estabilidad multisemilla obtenidos reproduciendo la metodología del Cuaderno 3 en `resultados_pca_kmeans.json`; resultados fijos sobre la base oficial. El análisis supervisado reajusta sus propias transformaciones exclusivamente en entrenamiento.
- **Bibliotecas solicitadas:** pandas, numpy, scikit-learn, matplotlib, seaborn y streamlit; se conserva Plotly para las visualizaciones interactivas.
- **Alcance:** el informe final, las fuentes, el análisis de faltantes y atípicos, la defensa individual y otros elementos de los entregables 1–4 y 6–7 se documentan en sus propios archivos. La aplicación no los sustituye.
