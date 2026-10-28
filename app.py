"""Proyecto de Analítica de Datos — Entregable 5 (Streamlit).
Reproduce el dataset de modelado y utiliza el transformador del Entregable 4
para ajustar transformaciones solamente con entrenamiento.
"""
from pathlib import Path
import hashlib
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    r2_score, mean_absolute_error, mean_squared_error,
)
from modelo_cuaderno4 import ExtractorPCAKMeans
from adaptador_datos import normalizar_datos, predictores_disponibles, calcular_segmentos, diagnostico_k, PREFIJOS

st.set_page_config(
    page_title='Analítica vial | Proyecto integrador',
    page_icon='📊',
    layout='wide',
)
BASE = Path(__file__).parent
RANDOM_STATE = 42

# Se conservan las variables originales cuando están disponibles.
# En CSV crudos/intermedios se usan únicamente los predictores observados.
COLUMNAS_REG = [
    'edad', 'sexo', 'politraumatismo', 'referido_a_otra_ips',
    'tipo_de_vehiculo', 'relacion_usuario_accidente', 'ips',
    'tipo_administradora', 'patologia_principal', 'region_principal',
    'mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos',
]
COLUMNAS_CLF = [c for c in COLUMNAS_REG if c != 'politraumatismo']


# ── Funciones de carga con caché ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def cargar_default():
    return pd.read_csv(BASE / 'data' / 'df_modelado.csv')


@st.cache_data(show_spinner=False)
def cargar_clusters():
    return pd.read_csv(BASE / 'data' / 'df_segmentado_clusters.csv')


@st.cache_data(show_spinner=False)
def cargar_base_ejemplo(nombre):
    rutas = {
        'Modelado oficial (28.276)': 'df_modelado.csv',
        'Base cruda original (34.306)': 'BBDDcruda_original.csv',
        'Base limpia intermedia (18.058)': 'BBDDlimpia.csv',
    }
    return pd.read_csv(BASE / 'data' / rutas[nombre], low_memory=False)


@st.cache_data(show_spinner=False)
def calcular_segmentos_cache(data):
    return calcular_segmentos(data)


@st.cache_data(show_spinner=False)
def diagnostico_k_cache(data):
    return diagnostico_k(data)


# ── Función de entrenamiento y evaluación ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def evaluar_modelo(data, objetivo, modelo, profundidad):
    """Reentrenamiento reproducible en la base suministrada.
    PCA/cluster nunca se leen de la exportación del Entregable 3.
    """
    pref = predictores_disponibles(data)
    columnas_base = ['edad', 'sexo', 'referido_a_otra_ips'] + list(pref)
    original = data.copy()
    original['edad'] = pd.to_numeric(original['edad'], errors='coerce')
    original = original.loc[original['edad'].notna() & original['politraumatismo'].isin(['si', 'no'])].copy()
    if objetivo == 'Costos':
        if 'costos' not in original.columns:
            raise ValueError('Para regresión falta la columna costos.')
        original['costos'] = pd.to_numeric(original['costos'], errors='coerce')
        original = original.loc[original['costos'].gt(0)].copy()
    if len(original) < 30:
        raise ValueError('No hay suficientes registros válidos para entrenar este modelo.')

    # La clasificación no exige un costo válido: evita excluir automáticamente
    # pacientes con costos nulos/cero si su respuesta binaria sí se conoce.
    y_clf = original['politraumatismo'].eq('si').astype(int)
    if y_clf.nunique() < 2 or y_clf.value_counts().min() < 2:
        raise ValueError('Se requieren al menos dos casos de cada clase de politraumatismo.')

    # Separación estratificada 80/20, semilla fija.
    ix_train, ix_test = train_test_split(
        np.arange(len(original)),
        test_size=0.20,
        stratify=y_clf.values,
        random_state=RANDOM_STATE,
    )

    # ── Regresión de costos ────────────────────────────────────────────────────
    if objetivo == 'Costos':
        y = np.log10(original['costos'].values)
        datos = original[columnas_base + ['politraumatismo']]
        configuracion = {
            'include_politrauma': True,
            'pref_dict': pref,
            'var_threshold': None if modelo == 'PCA completo + lineal' else 0.80,
        }
        pipe = Pipeline([
            ('pre', ExtractorPCAKMeans(**configuracion)),
            ('modelo', LinearRegression()),
        ])
        pipe.fit(datos.iloc[ix_train], y[ix_train])
        preditr = pipe.predict(datos.iloc[ix_train])
        predite = pipe.predict(datos.iloc[ix_test])
        # Factor corrector calculado SOLO con entrenamiento (Duan smearing).
        smearing = float(np.mean(np.power(10.0, y[ix_train] - preditr)))
        costo_pred = np.power(10.0, predite) * smearing
        return {
            'objetivo': objetivo, 'modelo': modelo,
            'y_test': y[ix_test], 'pred_log': predite,
            'costo_real': original['costos'].values[ix_test],
            'costo_predicho': costo_pred,
            'indices': original.index.to_numpy()[ix_test],
            'entrenamiento': len(ix_train), 'prueba': len(ix_test),
            'n_pc': pipe.named_steps['pre'].n_pcs_,
            'var_exp': pipe.named_steps['pre'].var_explicada_,
            'muestreo_svm': False, 'variables_modelo': columnas_base + ['politraumatismo'],
        }

    # ── Clasificación de politraumatismo ──────────────────────────────────────
    datos = original[columnas_base]
    y = y_clf.values
    subset_train = ix_train

    if modelo == 'Regresión logística equilibrada':
        estimator = LogisticRegression(
            class_weight='balanced', max_iter=1200, random_state=RANDOM_STATE
        )
    elif modelo == 'Regresión logística sin balanceo':
        estimator = LogisticRegression(max_iter=1200, random_state=RANDOM_STATE)
    elif modelo == 'Árbol de decisión':
        estimator = DecisionTreeClassifier(
            max_depth=profundidad, min_samples_leaf=25,
            class_weight='balanced', random_state=RANDOM_STATE,
        )
    elif modelo == 'Random Forest':
        estimator = RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_leaf=15,
            class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE,
        )
    elif modelo == 'SVM RBF (submuestra)':
        # Reducción explícita y reproducible; el conjunto de prueba no cambia.
        if len(ix_train) > 8000:
            local_ix, _ = train_test_split(
                ix_train, train_size=8000,
                stratify=y[ix_train], random_state=RANDOM_STATE,
            )
            subset_train = local_ix
        estimator = SVC(
            kernel='rbf', C=1.0, class_weight='balanced',
            probability=True, random_state=RANDOM_STATE,
        )
    else:
        raise ValueError('Modelo desconocido: ' + modelo)

    pipe = Pipeline([
        ('pre', ExtractorPCAKMeans(include_politrauma=False, var_threshold=0.80, pref_dict=pref)),
        ('modelo', estimator),
    ])
    pipe.fit(datos.iloc[subset_train], y[subset_train])
    prob = pipe.predict_proba(datos.iloc[ix_test])[:, 1]
    return {
        'objetivo': objetivo, 'modelo': modelo, 'probabilidades': prob,
        'y_test': y[ix_test], 'indices': original.index.to_numpy()[ix_test],
        'entrenamiento': len(subset_train), 'prueba': len(ix_test),
        'n_pc': pipe.named_steps['pre'].n_pcs_,
        'var_exp': pipe.named_steps['pre'].var_explicada_,
        'muestreo_svm': len(subset_train) != len(ix_train), 'variables_modelo': columnas_base,
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title('Analítica de Datos')
st.sidebar.caption('Accidentalidad vial y costos hospitalarios · Bucaramanga')
menu = st.sidebar.radio('Navegación', [
    'Inicio', 'Carga y calidad', 'Análisis exploratorio', 'PCA y K-means',
    'Regresión de costos', 'Clasificación', 'Reproducibilidad',
])
st.sidebar.divider()
base_activa = st.sidebar.selectbox('Base de datos incluida', [
    'Modelado oficial (28.276)',
    'Base cruda original (34.306)',
    'Base limpia intermedia (18.058)',
])
archivo = st.sidebar.file_uploader('O cargar un CSV propio (prioritario)', type=['csv'])

try:
    if archivo is not None:
        contenido = archivo.getvalue()
        if not contenido:
            raise ValueError('El archivo CSV está vacío.')
        # Detecta los dos delimitadores habituales sin alterar los datos de origen.
        linea = contenido.splitlines()[0].decode('utf-8-sig', errors='replace')
        sep = ';' if linea.count(';') > linea.count(',') else ','
        crudo = pd.read_csv(io.BytesIO(contenido), sep=sep, low_memory=False)
        procedencia = archivo.name
        archivo_personalizado = True
        firma_datos = hashlib.sha256(contenido).hexdigest()[:16]
    else:
        crudo = cargar_base_ejemplo(base_activa)
        procedencia = base_activa
        archivo_personalizado = base_activa != 'Modelado oficial (28.276)'
        firma_datos = 'ejemplo_' + hashlib.sha256(base_activa.encode()).hexdigest()[:16]
    if crudo.empty or len(crudo.columns) < 3:
        raise ValueError('El CSV está vacío o no tiene suficientes columnas. Comprueba el delimitador.')
    df, notas_esquema, esquema = normalizar_datos(crudo)
except Exception as exc:
    st.sidebar.error(f'No se pudo preparar el archivo: {exc}')
    st.stop()

st.sidebar.caption(f'Archivo: {procedencia} · Esquema: {esquema}')
with st.sidebar.expander('Compatibilidad y variables ausentes'):
    if notas_esquema:
        for nota in notas_esquema:
            st.write('• ' + nota)
    else:
        st.write('Variables del esquema de modelado oficial disponibles.')

st.sidebar.caption(f'Archivo activo: {len(df):,} filas × {len(df.columns)} columnas')


# ══════════════════════════════════════════════════════════════════════════════
# INICIO
# ══════════════════════════════════════════════════════════════════════════════
if menu == 'Inicio':
    st.title('Accidentalidad vial y costos de atención hospitalaria')
    st.markdown(
        '**Proyecto integrador de Analítica de Datos.** '
        'Herramienta exploratoria de apoyo al análisis operativo de datos '
        'hospitalarios históricos — Bucaramanga. Los resultados se calculan para la base activa.'
    )

    if not {'costos', 'politraumatismo'}.issubset(df.columns):
        st.warning('El CSV cargado no contiene las variables de los indicadores principales.')
    else:
        # KPIs principales
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric('Total atenciones', f'{len(df):,}')
        col_b.metric('Costo mediano', f"${df['costos'].median():,.0f} COP")
        col_c.metric('Costo promedio', f"${df['costos'].mean():,.0f} COP")
        col_d.metric(
            'Politraumatismo',
            f"{df['politraumatismo'].astype(str).str.lower().eq('si').mean()*100:.1f} %",
        )
        col_e.metric(
            'IPS distintas',
            str(df['ips'].nunique()) if 'ips' in df.columns else '—',
        )

        st.markdown('---')

        # Gráficos de resumen
        col_left, col_right = st.columns(2)
        with col_left:
            if 'ano' in df.columns:
                conteo_ano = (
                    df.groupby('ano').size()
                    .reset_index(name='Atenciones')
                )
                conteo_ano['ano_str'] = conteo_ano['ano'].astype(str)
                fig_ano = px.bar(
                    conteo_ano, x='ano_str', y='Atenciones',
                    title='Atenciones registradas por año',
                    labels={'ano_str': 'Año', 'Atenciones': 'Número de atenciones'},
                )
                fig_ano.update_traces(marker_color='#2563EB')
                fig_ano.update_layout(
                    showlegend=False,
                    xaxis=dict(type='category', title='Año'),
                    yaxis=dict(title='Número de atenciones'),
                )
                st.plotly_chart(fig_ano, use_container_width=True)

        with col_right:
            corte_p99 = df['costos'].quantile(0.99)
            costos_viz = df.loc[df['costos'] <= corte_p99, 'costos']
            fig_cos = px.histogram(
                costos_viz, nbins=50,
                title='Distribución de costos (hasta percentil 99)',
                labels={'value': 'Costo (COP)', 'count': 'Frecuencia'},
            )
            fig_cos.update_traces(marker_color='#2563EB')
            fig_cos.update_layout(
                showlegend=False,
                xaxis_title='Costo (COP)',
                yaxis_title='Frecuencia',
            )
            st.plotly_chart(fig_cos, use_container_width=True)

        # Prevalencia de politraumatismo por año
        if 'ano' in df.columns:
            st.markdown('#### Prevalencia de politraumatismo por año')
            es_politrauma = (
                df['politraumatismo'].astype(str).str.lower().eq('si').astype(float)
            )
            prev_ano = (
                df.assign(_pt=es_politrauma)
                .groupby('ano')['_pt']
                .mean()
                .mul(100)
                .round(2)
                .reset_index(name='Prevalencia (%)')
            )
            prev_ano['ano_str'] = prev_ano['ano'].astype(str)
            fig_prev = px.line(
                prev_ano, x='ano_str', y='Prevalencia (%)',
                markers=True,
                title='Prevalencia de politraumatismo por año',
                labels={'ano_str': 'Año', 'Prevalencia (%)': 'Prevalencia (%)'},
            )
            fig_prev.update_traces(line_color='#059669', marker_color='#059669')
            fig_prev.update_layout(
                xaxis=dict(type='category', title='Año'),
                yaxis=dict(title='Prevalencia (%)', ticksuffix='%'),
            )
            st.plotly_chart(fig_prev, use_container_width=True)

    st.markdown('---')
    col_info, col_uso = st.columns(2)
    with col_info:
        st.info(
            '**Metodología:** auditoría y EDA → PCA y segmentación → '
            'modelos supervisados → evaluación y decisiones. '
            'Las asociaciones no prueban causalidad.'
        )
    with col_uso:
        st.success(
            '**Cómo usar la aplicación:** use el menú lateral para cargar CSV, '
            'explorar variables, revisar los segmentos del Entregable 3 y '
            'entrenar modelos del Entregable 4 con partición fija 80/20.'
        )


# ══════════════════════════════════════════════════════════════════════════════
# CARGA Y CALIDAD
# ══════════════════════════════════════════════════════════════════════════════
elif menu == 'Carga y calidad':
    st.title('Carga y calidad de datos')
    st.caption('Los valores mostrados se calculan sobre el archivo seleccionado.')

    # Validación anticipada de columnas esenciales
    faltantes_esenciales = [c for c in ['costos', 'edad', 'politraumatismo'] if c not in df.columns]
    if faltantes_esenciales:
        st.error(
            f'El archivo no contiene columnas esenciales: '
            f'**{", ".join(faltantes_esenciales)}**. '
            'Carga un CSV compatible con la metodología del proyecto.'
        )

    a, b, c, d = st.columns(4)
    a.metric('Filas', f'{len(df):,}')
    b.metric('Variables', str(len(df.columns)))
    c.metric('Duplicados exactos', f'{df.duplicated().sum():,}')
    d.metric('Nulos explícitos', f'{int(df.isna().sum().sum()):,}')

    st.subheader('Primeras 10 filas')
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader('Información general del DataFrame (equivalente a df.info())')
    info_buffer = io.StringIO()
    df.info(buf=info_buffer, verbose=True, show_counts=True, memory_usage='deep')
    st.code(info_buffer.getvalue(), language='text')

    st.subheader('Resumen de las columnas')
    resumen = pd.DataFrame({
        'variable': df.columns,
        'tipo': df.dtypes.astype(str).values,
        'nulos': df.isna().sum().values,
        'porcentaje_nulos': (df.isna().mean() * 100).round(2).values,
        'valores_distintos': [df[c].nunique() for c in df.columns],
    })
    st.dataframe(resumen, use_container_width=True, hide_index=True)

    st.subheader('Faltantes encubiertos')
    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    encub = pd.DataFrame([{
        'variable': v,
        'sin_informacion_o_no_especificada_%': round(
            df[v].astype(str).isin(['sin_informacion', 'no_especificada']).mean() * 100, 2
        ),
    } for v in cat_cols])
    st.dataframe(
        encub.sort_values('sin_informacion_o_no_especificada_%', ascending=False),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        'Descargar auditoría CSV',
        resumen.to_csv(index=False).encode('utf-8-sig'),
        'auditoria_variables.csv', 'text/csv',
    )
    st.warning(
        'Que dos filas sean idénticas no demuestra que sean episodios duplicados; '
        'no se eliminan automáticamente.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS EXPLORATORIO
# ══════════════════════════════════════════════════════════════════════════════
elif menu == 'Análisis exploratorio':
    st.title('Análisis exploratorio (EDA)')
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # ── Filtros dinámicos ──────────────────────────────────────────────────────
    st.subheader('Filtros')
    fcol1, fcol2, fcol3 = st.columns(3)
    df_filtrado = df.copy()

    with fcol1:
        if 'ano' in df.columns:
            anos_disponibles = sorted(df['ano'].unique())
            anos_sel = st.multiselect('Año', anos_disponibles, default=list(anos_disponibles))
            if anos_sel:
                df_filtrado = df_filtrado[df_filtrado['ano'].isin(anos_sel)]

    with fcol2:
        if 'tipo_de_vehiculo' in df.columns:
            vehiculos = sorted(df['tipo_de_vehiculo'].dropna().unique())
            veh_sel = st.multiselect('Tipo de vehículo', vehiculos, default=list(vehiculos))
            if veh_sel:
                df_filtrado = df_filtrado[df_filtrado['tipo_de_vehiculo'].isin(veh_sel)]

    with fcol3:
        if 'relacion_usuario_accidente' in df.columns:
            roles = sorted(df['relacion_usuario_accidente'].dropna().unique())
            rol_sel = st.multiselect('Rol del usuario', roles, default=list(roles))
            if rol_sel:
                df_filtrado = df_filtrado[df_filtrado['relacion_usuario_accidente'].isin(rol_sel)]

    st.caption(f'Registros tras filtros: **{len(df_filtrado):,}** de {len(df):,}')
    st.markdown('---')

    # ── Estadísticas descriptivas ──────────────────────────────────────────────
    st.subheader('Estadísticas descriptivas')
    num_f = df_filtrado.select_dtypes(include=np.number).columns.tolist()
    if num_f:
        st.dataframe(df_filtrado[num_f].describe().T, use_container_width=True)

    izq, der = st.columns(2)
    with izq:
        if num_f:
            v = st.selectbox(
                'Variable numérica', num_f,
                index=num_f.index('costos') if 'costos' in num_f else 0,
            )
            graf = df_filtrado[[v]].dropna()
            if v == 'costos':
                corte = graf[v].quantile(.99)
                st.caption(
                    'Para visualizar mejor los costos se muestran valores '
                    'hasta el percentil 99; los datos originales no se recortan.'
                )
                graf = graf[graf[v] <= corte]
            fig_hist = px.histogram(
                graf, x=v, nbins=42, title='Distribución de ' + v,
                labels={'count': 'Frecuencia', v: v},
            )
            fig_hist.update_layout(yaxis_title='Frecuencia', xaxis_title=v)
            st.plotly_chart(fig_hist, use_container_width=True)

    with der:
        cat_f = df_filtrado.select_dtypes(include=['object', 'string']).columns.tolist()
        if cat_f:
            v_cat = st.selectbox(
                'Variable categórica', cat_f,
                index=cat_f.index('tipo_de_vehiculo') if 'tipo_de_vehiculo' in cat_f else 0,
            )
            conteos = (
                df_filtrado[v_cat].value_counts(dropna=False)
                .head(15).rename_axis(v_cat).reset_index(name='registros')
            )
            fig_bar = px.bar(
                conteos, x='registros', y=v_cat, orientation='h',
                title='Frecuencias de ' + v_cat,
                labels={'registros': 'Número de registros', v_cat: v_cat},
            )
            fig_bar.update_layout(xaxis_title='Número de registros', yaxis_title=v_cat)
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Matriz de correlaciones ────────────────────────────────────────────────
    if len(num_f) > 1:
        st.subheader('Matriz de correlaciones numéricas (Spearman)')
        vs = st.multiselect(
            'Variables de la matriz', num_f,
            default=[c for c in ['edad', 'costos', 'log_costos', 'ano', 'mes_numeric'] if c in num_f],
        )
        if len(vs) >= 2:
            cor = df_filtrado[vs].corr(method='spearman')
            grafico_corr = st.radio(
                'Visualización de correlaciones',
                ['Interactiva (Plotly)', 'Mapa de calor (Seaborn)'],
                horizontal=True,
            )
            if grafico_corr == 'Mapa de calor (Seaborn)':
                fig_cor, ax_cor = plt.subplots(figsize=(max(6, len(vs)*1.1), max(4, len(vs)*0.85)))
                sns.heatmap(cor, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1,
                            square=True, ax=ax_cor)
                ax_cor.set_title('Correlaciones de Spearman (Seaborn)')
                fig_cor.tight_layout()
                st.pyplot(fig_cor)
                plt.close(fig_cor)
            else:
                st.plotly_chart(
                    px.imshow(cor, text_auto='.2f', aspect='auto', zmin=-1, zmax=1,
                              color_continuous_scale='RdBu_r'),
                    use_container_width=True,
                )

    # ── Costos por IPS ─────────────────────────────────────────────────────────
    if {'ips', 'costos'}.issubset(df_filtrado.columns):
        st.subheader('Costos por IPS')
        comp = (
            df_filtrado.groupby('ips', dropna=False)
            .agg(casos=('costos', 'size'), mediana=('costos', 'median'))
            .reset_index()
        )
        st.dataframe(comp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PCA Y K-MEANS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == 'PCA y K-means':
    st.title('PCA y K-means: segmentación de la base activa')
    st.caption('Los componentes y clústeres se calculan para la base elegida, nunca se trasladan de otro archivo.')

    oficial = not archivo_personalizado
    if oficial:
        try:
            seg = cargar_clusters()
            from json import loads
            ruta_diag = BASE / 'resultados_pca_kmeans.json'
            diag_pca = loads(ruta_diag.read_text(encoding='utf-8')) if ruta_diag.exists() else None
            n_pca = diag_pca['n_pca_80'] if diag_pca else 11
            var_retenida = diag_pca['varianza_retenida'] if diag_pca else .8109
            st.caption('Resultados descriptivos precalculados del Entregable 3 sobre el dataset oficial.')
        except FileNotFoundError:
            oficial = False
            st.warning('No se encontró la exportación oficial. Se recalculará la segmentación.')
    if not oficial:
        st.info('Se realizará un nuevo PCA y K-means (k=3) únicamente con las variables disponibles en este archivo. '
                'La muestra y la especificación pueden diferir del Entregable 3; los resultados no son directamente comparables.')
        try:
            with st.spinner('Calculando PCA y K-means de la base activa...'):
                analisis = calcular_segmentos_cache(df)
            seg = analisis['segmentos']
            n_pca = analisis['n_pca_80']
            var_retenida = analisis['varianza_retenida']
            diag_pca = None
        except Exception as exc:
            st.error(f'No fue posible segmentar esta base: {exc}')
            st.stop()

    a,b,c,d = st.columns(4)
    a.metric('Registros segmentados', f'{len(seg):,}')
    b.metric('Grupos', str(seg['cluster'].nunique()))
    c.metric('Componentes retenidos', str(n_pca))
    d.metric('Varianza explicada', f'{var_retenida*100:.2f} %')
    st.warning('El PCA descriptivo puede incluir costos y politraumatismo. Sus componentes '
               'no se reutilizan como predictores supervisados de esos mismos resultados.')

    with st.expander('PCA: varianza explicada, cargas y selección de k', expanded=False):
        st.markdown('**PCA:** resume la variación mediante componentes ortogonales. '
                    '**K-means:** agrupa según distancias a centroides. Se usa k=3 para '
                    'permitir comparar perfiles, sin afirmar que sean categorías naturales.')
        if oficial and diag_pca:
            componentes = pd.DataFrame(diag_pca['variacion_componentes'])
            st.plotly_chart(px.line(componentes, x='componente', y='acumulada_pct', markers=True,
                labels={'componente':'Número de componentes','acumulada_pct':'Varianza acumulada (%)'},
                title='Varianza acumulada (dataset oficial)'), use_container_width=True)
            nombres = diag_pca['cargas_6']['variables']
            carga = pd.DataFrame(diag_pca['cargas_6']['componentes'], index=nombres)
            seleccion = pd.DataFrame(diag_pca['seleccion_k'])
        else:
            componentes = pd.DataFrame({'Componente': np.arange(1, len(analisis['varianza_acumulada'])+1),
                          'Varianza acumulada (%)': np.array(analisis['varianza_acumulada'])*100})
            st.plotly_chart(px.line(componentes, x='Componente', y='Varianza acumulada (%)', markers=True,
                title='Varianza acumulada (base seleccionada)'), use_container_width=True)
            carga = analisis['cargas']
            with st.spinner('Evaluando k=2 a 6 sobre una muestra (hasta 3.500 registros)...'):
                seleccion = diagnostico_k_cache(analisis)
            st.caption('Elbow y Silhouette son diagnósticos exploratorios calculados sobre una muestra reproducible; '
                       'los clústeres finales se ajustan con todas las filas válidas.')
        filas_carga = []
        for col in carga.columns[:min(3,len(carga.columns))]:
            for v in carga[col].abs().nlargest(4).index:
                filas_carga.append({'Componente':str(col), 'Variable':str(v), 'Carga':float(carga.loc[v,col])})
        st.dataframe(pd.DataFrame(filas_carga), use_container_width=True, hide_index=True)
        i,d = st.columns(2)
        with i:
            st.plotly_chart(px.line(seleccion, x='k', y='inercia', markers=True,
                title='Elbow: inercia según k'), use_container_width=True)
        with d:
            st.plotly_chart(px.line(seleccion, x='k', y='silhouette', markers=True,
                title='Silhouette según k'), use_container_width=True)
        st.dataframe(seleccion, use_container_width=True, hide_index=True)

    caps = (seg.groupby('cluster').agg(
        Registros=('edad', 'size'), Edad_mediana=('edad', 'median'),
        Costo_mediano_COP=('costos', 'median'), Costo_medio_COP=('costos', 'mean'),
        Politraumatismo_pct=('politraumatismo',lambda x:100*x.astype(str).str.lower().eq('si').mean()),
        Moto_pct=('tipo_de_vehiculo',lambda x:100*x.astype(str).eq('moto').mean()),
    ).round(2).reset_index())
    st.subheader('Perfiles de los tres clústeres')
    st.dataframe(caps, use_container_width=True, hide_index=True)
    max_puntos = min(12000, len(seg))
    if max_puntos > 500:
        n = st.slider('Puntos visibles en PC1 vs PC2', 500, max_puntos,
                      min(5000,max_puntos), 500)
    else:
        n = max_puntos
    mostrar = seg.sample(min(n,len(seg)), random_state=42).copy()
    mostrar['cluster'] = mostrar['cluster'].astype(str)
    st.plotly_chart(px.scatter(mostrar, x='PC1', y='PC2', color='cluster', opacity=.5,
        hover_data=['edad','costos','politraumatismo'],
        title='Proyección de clústeres en PC1 y PC2'), use_container_width=True)
    st.plotly_chart(px.bar(caps, x='cluster', y='Costo_mediano_COP',
        title='Costo mediano por clúster'),use_container_width=True)
    st.download_button('Descargar segmentos CSV',seg.to_csv(index=False).encode('utf-8-sig'),
        f'segmentos_{"oficial" if oficial else "base_activa"}.csv','text/csv')


# ══════════════════════════════════════════════════════════════════════════════
# REGRESIÓN DE COSTOS / CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif menu in ['Regresión de costos', 'Clasificación']:
    objetivo = 'Costos' if menu == 'Regresión de costos' else 'Politraumatismo'
    st.title('Regresión de costos' if objetivo == 'Costos' else 'Clasificación de politraumatismo')

    try:
        pref_activo = predictores_disponibles(df)
        if objetivo == 'Costos' and 'costos' not in df.columns:
            raise ValueError('Falta la variable objetivo costos.')
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.caption('Predictores categóricos disponibles para esta base: ' + ', '.join(pref_activo))
    if archivo_personalizado:
        st.warning('Se entrenará una especificación adaptada a las columnas reales de esta base. '
                   'No se incorporan predictores faltantes ni se comparan directamente sus métricas '
                   'con las del modelo oficial de 28.276 registros.')

    st.caption(
        'Separación estratificada 80/20 (semilla 42). '
        'El escalado, PCA y K-means se reajustan sobre entrenamiento '
        'y se aplican después a prueba.'
    )

    if objetivo == 'Costos':
        modelo = st.selectbox('Seleccionar especificación', ['PCA 80 % + lineal', 'PCA completo + lineal'])
        profundidad = 5
        st.info(
            'Respuesta continua: log₁₀(costos COP). '
            'Se muestra también error retransformado a COP con '
            'corrección de Duan, estimada sobre entrenamiento.'
        )
    else:
        modelo = st.selectbox('Seleccionar algoritmo', [
            'Regresión logística equilibrada',
            'Regresión logística sin balanceo',
            'Árbol de decisión',
            'Random Forest',
            'SVM RBF (submuestra)',
        ])
        profundidad = (
            st.slider('Profundidad máxima del árbol', 1, 15, 5)
            if modelo == 'Árbol de decisión'
            else 5
        )
        st.info(
            'Clase positiva: politraumatismo = sí. '
            'La prevalencia es minoritaria; evaluar accuracy sola puede inducir a error.'
        )

    explicaciones = {
        'PCA 80 % + lineal': (
            'Se reducen los predictores a componentes PCA que explican al menos el 80 % de la varianza '
            'de entrenamiento. Una regresión lineal estima log₁₀(costos) como combinación de esos componentes. '
            'La retransformación mediante Duan permite reportar también errores en pesos.'
        ),
        'PCA completo + lineal': (
            'Conserva todos los componentes PCA del preprocesamiento y ajusta una regresión lineal '
            'sobre log₁₀(costos). Sirve para contrastar la pérdida de información de la versión reducida; '
            'su mayor R² logarítmico no implica necesariamente menor RMSE en COP.'
        ),
        'Regresión logística equilibrada': (
            'Modela la probabilidad de politraumatismo mediante una función logística. '
            'Asigna mayor peso a la clase minoritaria durante el ajuste para atender el desbalance; '
            'esta decisión puede aumentar el recall a costa de falsas alarmas.'
        ),
        'Regresión logística sin balanceo': (
            'Estima probabilidades usando regresión logística, sin ponderar las clases. '
            'Al haber pocos positivos, puede priorizar la clase mayoritaria; se compara para evaluar '
            'el efecto del balanceo sobre precision, recall y F1.'
        ),
        'Árbol de decisión': (
            'Construye reglas de clasificación dividiendo sucesivamente los registros para disminuir '
            'la impureza Gini. La profundidad máxima y el mínimo de observaciones por hoja '
            'limitan la complejidad y ayudan a controlar el sobreajuste.'
        ),
        'Random Forest': (
            'Combina 120 árboles entrenados con muestras y subconjuntos aleatorios de variables. '
            'Promedia sus estimaciones de probabilidad y suele reducir la variabilidad de un único árbol. '
            'Sus resultados no demuestran por sí solos disponibilidad o causalidad de los predictores.'
        ),
        'SVM RBF (submuestra)': (
            'Busca una frontera que separe ambas clases en un espacio no lineal utilizando un kernel radial (RBF). '
            'El entrenamiento usa como máximo 8.000 registros estratificados para controlar el coste computacional; '
            'las probabilidades se calculan mediante el ajuste probabilístico interno de SVC.'
        ),
    }
    with st.expander('¿Cómo funciona este algoritmo y qué limitaciones tiene?', expanded=False):
        st.write(explicaciones[modelo])
        st.caption('El preprocesamiento se ajusta sobre entrenamiento. Las métricas se calculan sobre prueba; '
                   'son resultados descriptivos del modelo, no recomendaciones médicas automáticas.')

    # Cada resultado se asocia al contenido exacto del archivo, modelo y parámetros.
    cache_key = (firma_datos, objetivo, modelo, profundidad)
    if 'resultados_modelos' not in st.session_state:
        st.session_state['resultados_modelos'] = {}

    if st.button('Entrenar y evaluar sobre el 20 % de prueba', type='primary'):
        with st.spinner('Ajustando preprocesamiento y modelo sobre los datos de entrenamiento...'):
            try:
                st.session_state['resultados_modelos'][cache_key] = evaluar_modelo(
                    df, objetivo, modelo, profundidad
                )
            except Exception as exc:
                st.error(f'No fue posible entrenar el modelo.\nDetalle del error: {exc}')
                st.stop()

    r = st.session_state['resultados_modelos'].get(cache_key)
    if r is None:
        st.info(
            'Pulsa «Entrenar y evaluar sobre el 20 % de prueba» para calcular métricas verificables de este modelo en vivo. '
            'No se muestran resultados inventados.'
        )
    else:
        # ── Resultados de regresión ────────────────────────────────────────────────
        if objetivo == 'Costos':
            x1, x2, x3, x4 = st.columns(4)
            x1.metric('Train', f"{r['entrenamiento']:,}")
            x2.metric('Test', f"{r['prueba']:,}")
            x3.metric('Componentes PCA', str(r['n_pc']))
            x4.metric('Varianza PCA', f"{r['var_exp']*100:.1f} %")

            real = r['costo_real']
            pred = r['costo_predicho']
            mae_log  = mean_absolute_error(r['y_test'], r['pred_log'])
            rmse_log = float(np.sqrt(mean_squared_error(r['y_test'], r['pred_log'])))
            r2       = r2_score(r['y_test'], r['pred_log'])
            r2_cop   = r2_score(real, pred)
            mae_cop  = mean_absolute_error(real, pred)
            rmse_cop = float(np.sqrt(mean_squared_error(real, pred)))

            met1, met2, met3, met4 = st.columns(4)
            met1.metric('R² en test · log₁₀', f'{r2:.3f}')
            met2.metric('R² en test · COP', f'{r2_cop:.3f}')
            met3.metric('MAE en test (COP)', f'${mae_cop:,.0f}')
            met4.metric('RMSE en test (COP)', f'${rmse_cop:,.0f}')
            st.caption(
                f'Escala log₁₀: MAE = {mae_log:.3f} | RMSE = {rmse_log:.3f}. '
                'El R² logarítmico y el R² en COP miden comportamientos distintos; '
                'un R² en COP negativo indica peor ajuste cuadrático que predecir '
                'el costo medio del conjunto de prueba.'
            )

            # ── Explicación del R² y limitaciones predictivas ──────────────────────
            with st.expander('📊 Evaluación del R² y limitaciones predictivas'):
                if modelo == 'PCA 80 % + lineal':
                    st.markdown(f"""
**Especificación activa:** PCA con corte al 80 % de varianza ($m={r['n_pc']}$ componentes) + Regresión Lineal.  
**Métrica calculada en test:** **R² = {r2:.4f}** (explica únicamente el **{r2*100:.2f} %** de la varianza de $\\log_{{10}}(\\text{{costos}})$).

#### ¿Qué limita el rendimiento en esta especificación?
1. **Posible pérdida de información predictiva:** Retener componentes por varianza explicada no garantiza preservar las combinaciones más útiles para predecir costos. Para atribuir el cambio a componentes específicos harían falta análisis adicionales.
2. **Limitaciones predictivas críticas (no es adecuado para uso individual):**
   - El modelo deja el **{(1 - r2)*100:.1f} %** de la varianza sin explicar.
   - Con un MAE de **${mae_cop:,.0f} COP** y un RMSE de **${rmse_cop:,.0f} COP**, las predicciones individuales sufren discrepancias de cientos de miles o millones de pesos.
   - **Causa estructural en los datos:** Las variables disponibles al ingreso (edad, vehículo, aseguradora, IPS) no incluyen factores determinantes del gasto hospitalario (días en UCI, cirugías, material de osteosíntesis, medicamentos especiales).
   - Por ello, este bajo R² representa una limitación predictiva sustancial y no debe considerarse un desempeño satisfactorio para tarificación ni presupuestación médica.

#### Comparativa entre escalas logarítmica y pesos colombianos (COP)
| Métrica | Escala $\\log_{{10}}$ | Escala COP (con corrección de Duan) |
|---|---|---|
| **R² en test** | {r2:.4f} | {r2_cop:.4f} |
| **MAE** | {mae_log:.4f} unidades log | ${mae_cop:,.0f} COP |
| **RMSE** | {rmse_log:.4f} unidades log | ${rmse_cop:,.0f} COP |

- **Escala $\\log_{{10}}$:** Estabiliza la asimetría y el error relativo, pero atenúa la visibilidad de los desvíos monetarios reales.
- **Escala COP:** Refleja el impacto financiero directo. Aplica el factor de corrección de sesgo no lineal de Duan (*smearing*), calculado exclusivamente en entrenamiento para evitar fuga de información.
""")
                else:
                    st.markdown(f"""
**Especificación activa:** PCA completo ($m={r['n_pc']}$ componentes sin descarte) + Regresión Lineal.  
**Métrica calculada en test:** **R² = {r2:.4f}** (explica el **{r2*100:.2f} %** de la varianza de $\\log_{{10}}(\\text{{costos}})$).

#### Comparación con la especificación del 80%:
- Esta especificación retiene {r['n_pc']} componentes. Su **R² log₁₀ = {r2:.4f}** y su **R² en COP = {r2_cop:.4f}** deben contrastarse con los de la especificación PCA 80 % evaluada sobre exactamente la misma partición.
- En escala logarítmica, **el {(1 - r2)*100:.1f} % de la variación queda sin explicar**; esto no implica el mismo porcentaje en pesos.
- **Limitaciones predictivas:** A pesar de la ganancia respecto a la versión del 80 %, el error medio absoluto sigue siendo de **${mae_cop:,.0f} COP** y el RMSE de **${rmse_cop:,.0f} COP**. Esto confirma que las variables sociodemográficas y de transporte no alcanzan para modelar con precisión los costos clínicos individuales.

#### Comparativa entre escalas logarítmica y pesos colombianos (COP)
| Métrica | Escala $\\log_{{10}}$ | Escala COP (con corrección de Duan) |
|---|---|---|
| **R² en test** | {r2:.4f} | {r2_cop:.4f} |
| **MAE** | {mae_log:.4f} unidades log | ${mae_cop:,.0f} COP |
| **RMSE** | {rmse_log:.4f} unidades log | ${rmse_cop:,.0f} COP |
""")

            st.plotly_chart(
                px.scatter(
                    pd.DataFrame({'Costo real (COP)': real, 'Costo estimado (COP)': pred})
                    .sample(min(2500, len(real)), random_state=42),
                    x='Costo real (COP)', y='Costo estimado (COP)', opacity=.45,
                    title='Costo real y costo estimado (submuestra visual)',
                ),
                use_container_width=True,
            )

            export = pd.DataFrame({
                'indice_original': r['indices'],
                'costo_real_COP': real,
                'costo_estimado_COP': pred,
                'log_costo_estimado': r['pred_log'],
            })
            metricas_df = pd.DataFrame([{
                'modelo': modelo,
                'r2_test_log10': round(r2, 6),
                'r2_test_COP': round(r2_cop, 6),
                'mae_log10': round(mae_log, 6),
                'rmse_log10': round(rmse_log, 6),
                'mae_COP': round(mae_cop, 2),
                'rmse_COP': round(rmse_cop, 2),
                'n_train': r['entrenamiento'],
                'n_test': r['prueba'],
                'n_pca': r['n_pc'],
                'var_exp_pca': round(r['var_exp'], 6),
            }])
            col_dl1, col_dl2 = st.columns(2)
            col_dl1.download_button(
                '⬇ Descargar predicciones CSV',
                export.to_csv(index=False).encode('utf-8-sig'),
                'predicciones_costos.csv', 'text/csv',
            )
            col_dl2.download_button(
                '⬇ Descargar métricas CSV',
                metricas_df.to_csv(index=False).encode('utf-8-sig'),
                'metricas_regresion.csv', 'text/csv',
            )

        # ── Resultados de clasificación ────────────────────────────────────────────
        else:
            y    = np.asarray(r['y_test'])
            prob = np.asarray(r['probabilidades'])

            # ── 2. Resultados del modelo seleccionado (5 métricas clave) ──────────
            st.subheader('Resultados del modelo seleccionado')
            umbral_activo = float(st.session_state.get('umbral_clasificacion', 0.50))
            pred = (prob >= umbral_activo).astype(int)

            col_a, col_b, col_c, col_d, col_e = st.columns(5)
            col_a.metric('ROC-AUC', f'{roc_auc_score(y, prob):.3f}')
            col_b.metric('Average Precision (AP)', f'{average_precision_score(y, prob):.3f}')
            col_c.metric('Precision', f'{precision_score(y, pred, zero_division=0):.3f}')
            col_d.metric('Recall', f'{recall_score(y, pred, zero_division=0):.3f}')
            col_e.metric('F1-Score', f'{f1_score(y, pred, zero_division=0):.3f}')
            st.caption(
                f'Accuracy: {accuracy_score(y, pred):.3f} | '
                f'Train: {r["entrenamiento"]:,} | Test: {r["prueba"]:,} | '
                f'Componentes PCA: {r["n_pc"]} ({r["var_exp"]*100:.1f} % varianza acumulada) | '
                f'Casos positivos en test: {sum(y):,} ({100*np.mean(y):.2f} %)'
            )
            if r['muestreo_svm']:
                st.warning(
                    'El SVM usa una submuestra estratificada de hasta 8.000 registros de '
                    'entrenamiento para limitar recursos en Community Cloud. '
                    'Evalúa el mismo conjunto de prueba que otros modelos, pero no ha '
                    'usado el mismo tamaño de entrenamiento.'
                )

            # ── 3. Matriz de confusión con etiquetas claras ────────────────────────
            st.subheader('Matriz de confusión')
            umbral = st.slider(
                'Umbral de decisión (análisis de sensibilidad en test)',
                0.05, 0.95, umbral_activo, 0.05,
                key='umbral_clasificacion',
            )
            pred = (prob >= umbral).astype(int)
            cm_mat = confusion_matrix(y, pred, labels=[0, 1])
            tn, fp, fn, tp = cm_mat.ravel()

            fig_cm = px.imshow(
                cm_mat, text_auto=True, color_continuous_scale='Blues',
                labels={'x': 'Predicción del modelo', 'y': 'Valor real observado', 'color': 'Casos'},
                x=['No politraumatismo (0)', 'Politraumatismo (1)'],
                y=['No politraumatismo (0)', 'Politraumatismo (1)'],
                title=f'Matriz de confusión (umbral = {umbral:.2f})',
            )
            fig_cm.update_layout(xaxis_title='Predicción del modelo', yaxis_title='Valor real observado')
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption(
                f'Verdaderos negativos (TN): {tn:,} | Falsos positivos (FP): {fp:,} | '
                f'Falsos negativos (FN): {fn:,} | Verdaderos positivos (TP): {tp:,}. '
                f'Coste ponderado **hipotético** (5×FN + 1×FP) = {5*fn+fp:,} unidades arbitrarias; '
                'los pesos 5 y 1 son supuestos didácticos, no costos asistenciales reales.'
            )

            # ── 4. Curvas ROC y Precision–Recall ───────────────────────────────────
            st.subheader('Curvas ROC y Precision–Recall')
            col_roc, col_pr = st.columns(2)
            with col_roc:
                fpr, tpr, _ = roc_curve(y, prob)
                fig_roc = go.Figure()
                fig_roc.add_scatter(
                    x=fpr, y=tpr, name=f'Modelo (AUC {roc_auc_score(y, prob):.3f})',
                    line=dict(color='#2563EB', width=2),
                )
                fig_roc.add_scatter(
                    x=[0, 1], y=[0, 1], line={'dash': 'dash', 'color': 'gray'}, name='Azar (AUC 0,50)'
                )
                fig_roc.update_layout(
                    title='Curva ROC',
                    xaxis_title='Tasa de Falsos Positivos (FPR)',
                    yaxis_title='Tasa de Verdaderos Positivos (TPR)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_pr:
                pr_vals, rec_vals, _ = precision_recall_curve(y, prob)
                fig_pr = go.Figure()
                fig_pr.add_scatter(
                    x=rec_vals, y=pr_vals, name=f'Modelo (AP {average_precision_score(y, prob):.3f})',
                    line=dict(color='#059669', width=2),
                )
                fig_pr.add_hline(
                    y=np.mean(y), line_dash='dash', line_color='gray',
                    annotation_text=f'Prevalencia base ({np.mean(y)*100:.1f} %)',
                )
                fig_pr.update_layout(
                    title='Curva Precision–Recall',
                    xaxis_title='Recall (Sensibilidad)',
                    yaxis_title='Precision (Valor Predictivo Positivo)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                )
                st.plotly_chart(fig_pr, use_container_width=True)

            # Notas metodológicas sobre AP y umbrales clínicos
            with st.expander('ℹ️ Notas metodológicas sobre métricas de clasificación y umbral'):
                st.markdown(f"""
#### 1. Average Precision (AP) y Área Trapezoidal son medidas distintas
- La métrica reportada es **Average Precision (AP = {average_precision_score(y, prob):.4f})**, obtenida mediante `average_precision_score`. Representa la suma ponderada de precisiones al corte de cada variación en recall:
  $$\\text{{AP}} = \\sum_n (R_n - R_{{n-1}}) P_n$$
- **Diferencia técnica fundamental:** AP aproxima la integral exacta de la curva escalonada sin asumir linealidad entre puntos adyacentes. En contraste, el área calculada mediante la regla trapezoidal clásica (`auc(recall, precision)`) presupone conexiones rectas entre pares $(R, P)$, lo que puede producir un valor distinto al de AP. Ambas medidas responden a reglas de integración diferentes y deben identificarse por separado.

#### 2. Umbral de decisión (0,50) y advertencia sobre umbrales clínicos
- El umbral estándar de **0,50** es una referencia algorítmica para la comparación. **No demuestra que los costos de falsos positivos y falsos negativos sean iguales**, ni garantiza una decisión óptima sin probabilidades calibradas y costos estimados.
- **Asimetría de costos:** En urgencias y trauma vial, las consecuencias de un Falso Negativo (no clasificar a un paciente politraumatizado grave) difieren sustancialmente de un Falso Positivo (evaluación diagnóstica preventiva).
- **Advertencia metodológica:** El umbral de 0,50 **no ha sido optimizado con matrices de costo operativo real ni validado clínicamente**. En un entorno asistencial real, la selección de cualquier umbral operativo requiere una validación clínica prospectiva formal y un análisis institucional de costo-efectividad; no deben adoptarse umbrales empíricos sin dicha validación.
""")

            # ── 5. Tabla de métricas detalladas y botones de descarga ─────────────
            st.subheader('Tabla de métricas detalladas y descargas')
            informe = classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(informe).T, use_container_width=True)
            st.warning(
                'Cambiar el umbral conociendo los resultados de prueba NO equivale a validarlo. '
                'El umbral definitivo debe elegirse con validación sobre entrenamiento, '
                'por costos reales y revisión clínica.'
            )

            export = pd.DataFrame({
                'indice_original': r['indices'],
                'politraumatismo_real': y,
                'probabilidad_modelo': prob,
                'clasificacion_predicha': pred,
            })
            metricas_df = pd.DataFrame([{
                'modelo': modelo, 'umbral': umbral,
                'accuracy':  round(accuracy_score(y, pred), 6),
                'precision': round(precision_score(y, pred, zero_division=0), 6),
                'recall':    round(recall_score(y, pred, zero_division=0), 6),
                'f1':        round(f1_score(y, pred, zero_division=0), 6),
                'roc_auc':   round(roc_auc_score(y, prob), 6),
                'average_precision_ap': round(average_precision_score(y, prob), 6),
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                'n_train': r['entrenamiento'], 'n_test': r['prueba'],
                'n_pca': r['n_pc'], 'var_exp_pca': round(r['var_exp'], 6),
            }])
            col_dl1, col_dl2 = st.columns(2)
            col_dl1.download_button(
                '⬇ Descargar predicciones CSV',
                export.to_csv(index=False).encode('utf-8-sig'),
                'predicciones_politraumatismo.csv', 'text/csv',
            )
            col_dl2.download_button(
                '⬇ Descargar métricas CSV',
                metricas_df.to_csv(index=False).encode('utf-8-sig'),
                'metricas_clasificacion.csv', 'text/csv',
            )

        # ── Comparación longitudinal de modelos ejecutados en la sesión ────────────
        resumenes = []
        for (firma, obj, nombre, prof), resultado in st.session_state['resultados_modelos'].items():
            if firma != firma_datos or obj != objetivo:
                continue
            if obj == 'Costos':
                resumenes.append({
                    'Modelo': nombre,
                    'Train': resultado['entrenamiento'],
                    'Test': resultado['prueba'],
                    'R² log₁₀': round(r2_score(resultado['y_test'], resultado['pred_log']), 4),
                    'R² COP': round(r2_score(resultado['costo_real'], resultado['costo_predicho']), 4),
                    'MAE COP': round(mean_absolute_error(resultado['costo_real'], resultado['costo_predicho'])),
                    'RMSE COP': round(np.sqrt(mean_squared_error(resultado['costo_real'], resultado['costo_predicho']))),
                })
            else:
                yy  = np.asarray(resultado['y_test'])
                pp  = np.asarray(resultado['probabilidades'])
                clas = (pp >= 0.50).astype(int)
                resumenes.append({
                    'Modelo': nombre + (f' · profundidad {prof}' if nombre == 'Árbol de decisión' else ''),
                    'Train': resultado['entrenamiento'],
                    'Test': resultado['prueba'],
                    'ROC-AUC': round(roc_auc_score(yy, pp), 4),
                    'Average Precision (AP)': round(average_precision_score(yy, pp), 4),
                    'Recall (umbral 0,5)': round(recall_score(yy, clas, zero_division=0), 4),
                    'F1 (umbral 0,5)':     round(f1_score(yy, clas, zero_division=0), 4),
                })

        if len(resumenes) > 1:
            st.subheader('Comparación de modelos ejecutados en esta sesión')
            st.caption(
                'Resultados sobre la misma partición de prueba (semilla 42). '
                'Para clasificación se usa un umbral común de 0,50 en esta tabla. '
                'El SVM usa menos filas de entrenamiento.'
            )
            resumen_comp = pd.DataFrame(resumenes)
            st.dataframe(resumen_comp, use_container_width=True, hide_index=True)
            st.download_button(
                '⬇ Descargar comparación de modelos CSV',
                resumen_comp.to_csv(index=False).encode('utf-8-sig'),
                f'comparacion_modelos_{"costos" if objetivo == "Costos" else "clasificacion"}.csv',
                'text/csv',
            )

    # ── 6. Sección desplegable: Análisis de sensibilidad y posible fuga de información ──
    if objetivo == 'Politraumatismo':
        if not archivo_personalizado:
            st.markdown('---')
            with st.expander('Análisis de sensibilidad y posible fuga de información', expanded=(r is None)):
                st.caption(
                    'Evaluación de sensibilidad experimental comparando los clasificadores originales '
                    '(13 variables, incluyendo patología principal y región principal) frente a una versión reducida '
                    '(11 variables, excluyendo ambas variables diagnósticas). '
                    'Resultados sobre el dataset oficial de modelado (28.276 registros, semilla 42, partición 80/20).'
                )

                path_exp = BASE / 'fase2_experimento_comparativo_completo.csv'
                if path_exp.exists():
                    df_exp = pd.read_csv(path_exp)

                    # Gráficos comparativos
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        fig_roc_comp = px.bar(
                            df_exp, x='modelo', y='roc_auc', color='variante',
                            barmode='group',
                            title='Comparación ROC-AUC (Original vs. Reducido)',
                            labels={'roc_auc': 'ROC-AUC', 'modelo': 'Modelo', 'variante': 'Especificación'},
                            color_discrete_map={
                                'Original (con patología y región)': '#2563EB',
                                'Reducido (sin patología ni región)': '#DC2626',
                            },
                        )
                        fig_roc_comp.add_hline(y=0.50, line_dash='dash', line_color='gray', annotation_text='Azar (0,50)')
                        fig_roc_comp.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                        st.plotly_chart(fig_roc_comp, use_container_width=True)

                    with col_g2:
                        fig_ap_comp = px.bar(
                            df_exp, x='modelo', y='pr_auc', color='variante',
                            barmode='group',
                            title='Comparación Average Precision (AP)',
                            labels={'pr_auc': 'Average Precision (AP)', 'modelo': 'Modelo', 'variante': 'Especificación'},
                            color_discrete_map={
                                'Original (con patología y región)': '#2563EB',
                                'Reducido (sin patología ni región)': '#DC2626',
                            },
                        )
                        fig_ap_comp.add_hline(y=0.1034, line_dash='dash', line_color='gray', annotation_text='Prevalencia base (10,3 %)')
                        fig_ap_comp.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                        st.plotly_chart(fig_ap_comp, use_container_width=True)

                    # Tabla detallada de métricas y matrices de confusión
                    st.markdown('##### Métricas detalladas y matrices de confusión (umbral 0,50)')
                    cols_mostrar = [
                        'modelo', 'variante', 'roc_auc', 'pr_auc',
                        'precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp'
                    ]
                    df_mostrar = df_exp[cols_mostrar].copy()
                    df_mostrar['matriz_confusion_[TN,FP,FN,TP]'] = df_mostrar.apply(
                        lambda row: f"[{int(row['tn'])}, {int(row['fp'])}, {int(row['fn'])}, {int(row['tp'])}]", axis=1
                    )
                    df_mostrar = df_mostrar.drop(columns=['tn', 'fp', 'fn', 'tp'])
                    df_mostrar.columns = [
                        'Modelo', 'Variante', 'ROC-AUC', 'Average Precision (AP)',
                        'Precision', 'Recall', 'F1-Score', 'Matriz Confusión [TN, FP, FN, TP]'
                    ]
                    st.dataframe(df_mostrar, use_container_width=True, hide_index=True)

                    # Explicación del impacto y advertencia metodológica
                    st.info(
                        '**Impacto de excluir `patologia_principal` y `region_principal`:**\n'
                        '- **ROC-AUC:** Cae de ~0,96 a ~0,62 (pérdida de más de 33 puntos de capacidad discriminativa).\n'
                        '- **Average Precision (AP):** Se reduce de ~0,70 a ~0,14 (muy cercano a la línea base aleatoria del 10,34 % de prevalencia).\n'
                        '- **Falsos positivos:** Se multiplican significativamente (en Random Forest pasan de 328 a 2.711 falsos positivos a umbral 0,50).\n\n'
                        '**Aclaración metodológica sobre fuga temporal:** Estos resultados son retrospectivos. La caída en el rendimiento por sí sola '
                        '**no demuestra matemáticamente fuga temporal**; lo que la sustenta es la verificación del flujo clínico y operativo: en el momento '
                        'previsto de predicción temprana (triaje en urgencias o en vía pública), el diagnóstico codificado (CIE-10) y la región anatómica confirmada '
                        '**aún no han sido determinados**. Incluirlas en un modelo predictivo temprano contamina el análisis con información retrospectiva del egreso.'
                    )

                    # Botón de descarga CSV
                    st.download_button(
                        '⬇ Descargar experimento completo de sensibilidad en CSV',
                        path_exp.read_bytes(),
                        'fase2_experimento_comparativo_completo.csv',
                        'text/csv',
                    )

    st.caption(
        'Los algoritmos se utilizan como ejercicios de análisis retrospectivo. '
        'No se recomienda su uso para decisiones médicas o triaje automático '
        'sin validación externa.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════
elif menu == 'Reproducibilidad':
    st.title('Reproducibilidad y limitaciones')
    st.write(
        'El archivo `requirements.txt` documenta bibliotecas; '
        '`modelo_cuaderno4.py` integra el transformador de la metodología del grupo; '
        'la fuente del modelado se mantiene sin modificar.'
    )
    st.code(
        'python -m venv .venv\n'
        '.venv\\Scripts\\activate\n'
        'pip install -r requirements.txt\n'
        'streamlit run app.py',
        language='bash',
    )
    st.write(
        '**Parámetros fijos:** 80 % entrenamiento; 20 % prueba; estratificación binaria; '
        'semilla 42; PCA 80 % salvo modelo completo; K-means k=3. '
        'SVM usa hasta 8.000 filas de train.'
    )
    st.write(
        '**Advertencias:** no usar `PC1`, `PC2`, `cluster` del archivo exportado como '
        'predictores para objetivos con fuga de información; verificar disponibilidad '
        'temporal de diagnósticos; no eliminar duplicados sin comprender la unidad de atención.'
    )
    st.write(
        '**Despliegue y privacidad:** El equipo indicó que los CSV incluidos son muestras ficticias '
        'para demostración académica. Esa condición no ha sido verificada de manera independiente; '
        'confirma procedencia, licencia y ausencia de datos personales antes de compartir el repositorio. '
        'Si se emplean datos reales, se requieren autorización y controles de privacidad.'
    )
    st.markdown(
        '**Fuente del proyecto:** Ficha comparativa E1; entregables 1–4; '
        'bases de datos provistas por el equipo.'
    )
